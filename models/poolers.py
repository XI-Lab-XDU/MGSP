import warnings
from typing import Optional, List, Dict, Tuple, Union

import torch
import torch.fx
import torchvision
from torch import nn, Tensor
import torch.nn.functional as F
from torchvision.ops.boxes import box_area

# from ..utils import _log_api_usage_once
# from .roi_align import roi_align
from torchvision.ops import roi_align
from torchvision.utils import _log_api_usage_once
from mmcv.ops import DeformRoIPool, deform_roi_pool


# copying result_idx_in_level to a specific index in result[]
# is not supported by ONNX tracing yet.
# _onnx_merge_levels() is an implementation supported by ONNX
# that merges the levels to the right indices
@torch.jit.unused
def _onnx_merge_levels(levels: Tensor, unmerged_results: List[Tensor]) -> Tensor:
    first_result = unmerged_results[0]
    dtype, device = first_result.dtype, first_result.device
    res = torch.zeros(
        (levels.size(0), first_result.size(1), first_result.size(2), first_result.size(3)), dtype=dtype, device=device
    )
    for level in range(len(unmerged_results)):
        index = torch.where(levels == level)[0].view(-1, 1, 1, 1)
        index = index.expand(
            index.size(0),
            unmerged_results[level].size(1),
            unmerged_results[level].size(2),
            unmerged_results[level].size(3),
        )
        res = res.scatter(0, index, unmerged_results[level])
    return res


# TODO: (eellison) T54974082 https://github.com/pytorch/pytorch/issues/26744/pytorch/issues/26744
def initLevelMapper(
    k_min: int,
    k_max: int,
    canonical_scale: int = 224,
    canonical_level: int = 4,
    eps: float = 1e-6,
):
    return LevelMapper(k_min, k_max, canonical_scale, canonical_level, eps)


class LevelMapper:
    """Determine which FPN level each RoI in a set of RoIs should map to based
    on the heuristic in the FPN paper.

    Args:
        k_min (int)
        k_max (int)
        canonical_scale (int)
        canonical_level (int)
        eps (float)
    """

    def __init__(
        self,
        k_min: int,
        k_max: int,
        canonical_scale: int = 224,
        canonical_level: int = 4,
        eps: float = 1e-6,
    ):
        self.k_min = k_min
        self.k_max = k_max
        self.s0 = canonical_scale
        self.lvl0 = canonical_level
        self.eps = eps

    def __call__(self, boxlists: List[Tensor]) -> Tensor:
        """
        Args:
            boxlists (list[BoxList])
        """
        # Compute level ids
        s = torch.sqrt(torch.cat([box_area(boxlist) for boxlist in boxlists]))

        # Eqn.(1) in FPN paper
        target_lvls = torch.floor(self.lvl0 + torch.log2(s / self.s0) + torch.tensor(self.eps, dtype=s.dtype))
        target_lvls = torch.clamp(target_lvls, min=self.k_min, max=self.k_max)
        return (target_lvls.to(torch.int64) - self.k_min).to(torch.int64)


def _convert_to_roi_format(boxes: List[Tensor]) -> Tensor:
    concat_boxes = torch.cat(boxes, dim=0)
    device, dtype = concat_boxes.device, concat_boxes.dtype
    ids = torch.cat(
        [torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device) for i, b in enumerate(boxes)],
        dim=0,
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def _infer_scale(feature: Tensor, original_size: List[int]) -> float:
    # assumption: the scale is of the form 2 ** (-k), with k integer
    size = feature.shape[-2:]
    possible_scales: List[float] = []
    for s1, s2 in zip(size, original_size):
        approx_scale = float(s1) / float(s2)
        scale = 2 ** float(torch.tensor(approx_scale).log2().round())
        possible_scales.append(scale)
    return possible_scales[0]


def _setup_scales(
    features: List[Tensor], image_shapes: List[Tuple[int, int]], canonical_scale: int, canonical_level: int
) -> Tuple[List[float], LevelMapper]:
    if not image_shapes:
        raise ValueError("images list should not be empty")
    max_x = 0
    max_y = 0
    for shape in image_shapes:
        max_x = max(shape[0], max_x)
        max_y = max(shape[1], max_y)
    original_input_shape = (max_x, max_y)

    scales = [_infer_scale(feat, original_input_shape) for feat in features]
    # get the levels in the feature map by leveraging the fact that the network always
    # downsamples by a factor of 2 at each level.
    lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
    lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()

    map_levels = initLevelMapper(
        int(lvl_min),
        int(lvl_max),
        canonical_scale=canonical_scale,
        canonical_level=canonical_level,
    )
    return scales, map_levels


def _filter_input(x: Dict[str, Tensor], featmap_names: List[str]) -> List[Tensor]:
    x_filtered = []
    for k, v in x.items():
        if k in featmap_names:
            x_filtered.append(v)
    return x_filtered


class DeformRoIPoolPack(DeformRoIPool):

    def __init__(self,
                 output_size: Tuple[int, ...],
                 output_channels: int,
                 deform_fc_channels: int = 1024,
                 gamma: float = 0.1):
        super().__init__(output_size, gamma)

        self.output_channels = output_channels
        self.deform_fc_channels = deform_fc_channels

        self.offset_fc = nn.Sequential(
            nn.Linear(
                self.output_size[0] * self.output_size[1] *
                self.output_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels, self.deform_fc_channels),
            nn.ReLU(inplace=True),
            nn.Linear(self.deform_fc_channels,
                      self.output_size[0] * self.output_size[1] * 2))
        self.offset_fc[-1].weight.data.zero_()
        self.offset_fc[-1].bias.data.zero_()

    def forward(self, input: Tensor, rois: Tensor, spatial_scale, sampling_ratio) -> Tensor:  # type: ignore
        assert input.size(1) == self.output_channels
        x = deform_roi_pool(input, rois, None, self.output_size,
                            spatial_scale, sampling_ratio,
                            self.gamma)
        rois_num = rois.size(0)
        offset = self.offset_fc(x.view(rois_num, -1))
        offset = offset.view(rois_num, 2, self.output_size[0],
                             self.output_size[1])
        return deform_roi_pool(input, rois, offset, self.output_size,
                               spatial_scale, sampling_ratio,
                               self.gamma)


class MultiScaleRoIAlign(nn.Module):
    """
    Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.

    It infers the scale of the pooling via the heuristics specified in eq. 1
    of the `Feature Pyramid Network paper <https://arxiv.org/abs/1612.03144>`_.
    They keyword-only parameters ``canonical_scale`` and ``canonical_level``
    correspond respectively to ``224`` and ``k0=4`` in eq. 1, and
    have the following meaning: ``canonical_level`` is the target level of the pyramid from
    which to pool a region of interest with ``w x h = canonical_scale x canonical_scale``.

    Args:
        featmap_names (List[str]): the names of the feature maps that will be used
            for the pooling.
        output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
        sampling_ratio (int): sampling ratio for ROIAlign
        canonical_scale (int, optional): canonical_scale for LevelMapper
        canonical_level (int, optional): canonical_level for LevelMapper

    Examples::

        >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
        >>> i = OrderedDict()
        >>> i['feat1'] = torch.rand(1, 5, 64, 64)
        >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
        >>> i['feat3'] = torch.rand(1, 5, 16, 16)
        >>> # create some random bounding boxes
        >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
        >>> # original image size, before computing the feature maps
        >>> image_sizes = [(512, 512)]
        >>> output = m(i, [boxes], image_sizes)
        >>> print(output.shape)
        >>> torch.Size([6, 5, 3, 3])

    """

    __annotations__ = {"scales": Optional[List[float]], "map_levels": Optional[LevelMapper]}

    def __init__(
        self,
        featmap_names: List[str],
        output_size: Union[int, Tuple[int], List[int]],
        sampling_ratio: int,
        *,
        canonical_scale: int = 224,
        canonical_level: int = 4,
    ):
        super().__init__()
        _log_api_usage_once(self)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.featmap_names = featmap_names
        self.sampling_ratio = sampling_ratio
        self.output_size = tuple(output_size)
        self.scales = None
        self.map_levels = None
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        self.poolers = nn.ModuleList()
        # self.rescalers = nn.ModuleList()
        
        for i in range(len(self.featmap_names)):
            pooler = DeformRoIPoolPack(output_size=self.output_size, output_channels=1024)
            self.poolers.append(pooler)
            # rescaler = nn.BatchNorm1d(1, affine=True)
            # self.rescalers.append(rescaler)
        
        

    def convert_to_roi_format(self, boxes: List[Tensor]) -> Tensor:
        warnings.warn("The 'convert_to_roi_format' method is deprecated since 0.12 and will be removed in 0.14.")
        return _convert_to_roi_format(boxes)

    def infer_scale(self, feature: Tensor, original_size: List[int]) -> float:
        warnings.warn("The 'infer_scale' method is deprecated since 0.12 and will be removed in 0.14.")
        return _infer_scale(feature, original_size)

    def setup_setup_scales(
        self,
        features: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> None:
        warnings.warn("The 'setup_setup_scales' method is deprecated since 0.12 and will be removed in 0.14.")
        self.scales, self.map_levels = _setup_scales(features, image_shapes, self.canonical_scale, self.canonical_level)

    def pooler_per_level(
        self, 
        level,
        input: Tensor, 
        rois: Tensor, 
        spatial_scale, 
        sampling_ratio
    ):
        result_level = self.poolers[level](
                        input,
                        rois,
                        spatial_scale=spatial_scale,
                        sampling_ratio=sampling_ratio,
                    )
        
        result_level_max_pool = F.adaptive_max_pool2d(result_level, 1).flatten(1)

        norms = result_level_max_pool.norm(2, 1, keepdim=True)
        
        norms = self.rescalers[level](norms)
        
        return result_level, norms
        
    
    def forward(
        self,
        x: Dict[str, Tensor],
        boxes: List[Tensor],
        image_shapes: List[Tuple[int, int]],
    ) -> Tensor:
        """
        Args:
            x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
                all the same number of channels, but they can have different sizes.
            boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
                (x1, y1, x2, y2) format and in the image reference size, not the feature map
                reference. The coordinate must satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
            image_shapes (List[Tuple[height, width]]): the sizes of each image before they
                have been fed to a CNN to obtain feature maps. This allows us to infer the
                scale factor for each one of the levels to be pooled.
        Returns:
            result (Tensor)
        """
        x_filtered = _filter_input(x, self.featmap_names)
        if self.scales is None or self.map_levels is None:
            self.scales, self.map_levels = _setup_scales(
                x_filtered, image_shapes, self.canonical_scale, self.canonical_level
            )

        return self._multiscale_roi_align(
            x_filtered,
            boxes,
            self.output_size,
            self.sampling_ratio,
            self.scales,
            self.map_levels,
        )
        
    def _multiscale_roi_align(
        self,
        x_filtered: List[Tensor],
        boxes: List[Tensor],
        output_size: List[int],
        sampling_ratio: int,
        scales: Optional[List[float]],
        mapper: Optional[LevelMapper],
    ) -> Tensor:

        if scales is None or mapper is None:
            raise ValueError("scales and mapper should not be None")

        num_levels = len(x_filtered)
        rois = _convert_to_roi_format(boxes)

        if num_levels == 1:
            return self.poolers[0](
                x_filtered[0],
                rois,
                # output_size=output_size,
                spatial_scale=scales[0],
                sampling_ratio=sampling_ratio,
            )

        levels = mapper(boxes)

        num_rois = len(rois)
        num_channels = x_filtered[0].shape[1]

        dtype, device = x_filtered[0].dtype, x_filtered[0].device
        result = []
        norms = []

        tracing_results = []
        for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
            idx_in_level = torch.where(levels == level)[0]
            rois_per_level = rois[idx_in_level]

            # result_idx_in_level = roi_align(
            #     per_level_feature,
            #     rois,
            #     output_size=output_size,
            #     spatial_scale=scale,
            #     sampling_ratio=sampling_ratio,
            # )
            result_idx_in_level, norm = self.pooler_per_level(
                level,
                per_level_feature,
                rois,
                scale,
                sampling_ratio
            )

            if torchvision._is_tracing():
                tracing_results.append(result_idx_in_level.to(dtype))
            else:
                # result and result_idx_in_level's dtypes are based on dtypes of different
                # elements in x_filtered.  x_filtered contains tensors output by different
                # layers.  When autocast is active, it may choose different dtypes for
                # different layers' outputs.  Therefore, we defensively match result's dtype
                # before copying elements from result_idx_in_level in the following op.
                # We need to cast manually (can't rely on autocast to cast for us) because
                # the op acts on result in-place, and autocast only affects out-of-place ops.
                result.append(result_idx_in_level.to(dtype))
            norms.append(norm)
        if torchvision._is_tracing():
            result = _onnx_merge_levels(levels, tracing_results)

        norms = torch.concat(norms, dim=-1)
        norms = F.softmax(norms, dim=-1)
        b, l = norms.shape
        weights = norms.view(b, l, 1, 1, 1)
        features_stack = torch.stack(result, dim=1)
        weighted_features = features_stack * weights
        aggregated_features = weighted_features.sum(dim=1)
        
        return aggregated_features


    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(featmap_names={self.featmap_names}, "
            f"output_size={self.output_size}, sampling_ratio={self.sampling_ratio})"
        )
