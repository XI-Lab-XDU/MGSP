# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

from copy import deepcopy
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops
from torchvision.models.detection import _utils as det_utils

from loss.oim import OIMLoss, OIMLossSafe
from loss.softmax_loss import SoftmaxLoss
from models.resnet import build_resnet, build_clip_resnet, load_clip_to_cpu, PromptLearner, TextEncoder, build_convext, build_vit
from models.transformer import TransformerHead
from .resnet import SimpleFeaturePyramid
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.layers import ShapeSpec
import json
# from .clip import clip
import open_clip
from scipy.linalg import hadamard
import os
from typing import Dict, List, Optional, Tuple
from utils.rpn_utils import ImageList, concat_box_prediction_layers
from torch import nn, Tensor
import numpy as np
from torch.autograd import Function
os.environ['HUGGINGFACE_HUB_CACHE'] = '/home/kpn/.cache/huggingface/hub'


class PrototypeGuidedRPN(RegionProposalNetwork):
    """
    Spatial-Level Prior Guided-RPN
    """
    def __init__(self, *args, backbone_out_dim=1024, num_prototypes=8, **kwargs):
        super().__init__(*args, **kwargs)

        self.backbone_out_dim = backbone_out_dim
        self.num_prototypes = num_prototypes
        self.proto_dim = 256

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, 1, 14, 14) * 0.01)
        self.prototype_keys = nn.Parameter(torch.randn(num_prototypes, self.proto_dim))

        self.visual_projector = nn.Sequential(
            nn.Linear(backbone_out_dim, self.proto_dim),
            nn.LayerNorm(self.proto_dim),
            nn.ReLU(),
            nn.Linear(self.proto_dim, self.proto_dim)
        )

        self.text_projector = nn.Sequential(
            nn.Linear(1024, self.proto_dim),
            nn.LayerNorm(self.proto_dim),
            nn.ReLU(),
            nn.Linear(self.proto_dim, self.proto_dim)
        )

        self.gate_conv = nn.Conv2d(1025, self.head.cls_logits.out_channels, 3, padding=1)

        nn.init.xavier_uniform_(self.gate_conv.weight)
        nn.init.zeros_(self.gate_conv.bias)
        nn.init.xavier_uniform_(self.visual_projector[0].weight)
        nn.init.xavier_uniform_(self.text_projector[0].weight)

    def compute_loss(self, objectness, pred_bbox_deltas, labels, regression_targets):
        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]
        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)
        objectness = objectness.flatten()
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = (
            F.smooth_l1_loss(
                pred_bbox_deltas[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1 / 9,
                reduction="none",
            )
            / (sampled_inds.numel())
        ).sum()

        _objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds], reduction='none')
        objectness_loss = (_objectness_loss / _objectness_loss.size(0)).sum()

        return objectness_loss, box_loss

    def forward(
        self,
        images: ImageList,
        features: Dict[str, Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None,
        current_epoch: int = 0
    ) -> Tuple[List[Tensor], Dict[str, Tensor]]:

        features = list(features.values())
        F_img = features[0]
        B, C, H, W = F_img.shape

        warmup_epochs = 5.0

        if self.training:
            text_weight = max(0.0, 1.0 - current_epoch / warmup_epochs)
        else:
            text_weight = 0.0

        v_global = F.adaptive_avg_pool2d(F_img, (1, 1)).flatten(1)
        q_v = F.normalize(self.visual_projector(v_global), dim=1)
        keys = F.normalize(self.prototype_keys, dim=1)

        logits_v = torch.mm(q_v, keys.t()) * 10.0
        weights_v = F.softmax(logits_v, dim=1)

        H_proto = None
        loss_consistency = torch.tensor(0.0, device=F_img.device)

        if self.training:
            if targets is None: raise ValueError("Targets needed for training")

            clip_text_feats = torch.cat([t["global_text_features"] for t in targets], dim=0)
            q_t = F.normalize(self.text_projector(clip_text_feats), dim=1)

            loss_align_rpn = (
                F.cross_entropy((q_v @ q_t.t()) * 10, torch.arange(B, device=q_v.device)) +
                F.cross_entropy((q_t @ q_v.t()) * 10, torch.arange(B, device=q_v.device))
             ) / 2

            logits_t = torch.mm(q_t, keys.t()) * 10.0
            weights_t = F.softmax(logits_t, dim=1)

            loss_consistency = F.kl_div(
                F.log_softmax(logits_v, dim=1),
                weights_t.detach(),
                reduction='batchmean'
            ) * 5.0

            weights_final = text_weight * weights_t + (1 - text_weight) * weights_v

            selected_proto = torch.einsum('bk,kchw->bchw', weights_final, self.prototypes)

        else:
            selected_proto = torch.einsum('bk,kchw->bchw', weights_v, self.prototypes)

        H_proto = F.interpolate(selected_proto, size=(H, W), mode='bilinear', align_corners=False)
        H_proto = torch.sigmoid(H_proto)

        objectness, pred_bbox_deltas = self.head(features)

        gate_input = torch.cat([F_img, H_proto], dim=1)

        G = torch.sigmoid(self.gate_conv(gate_input))

        objectness[0] = objectness[0] * (1 + 0.3 * G)

        anchors = self.anchor_generator(images, features)
        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        proposals = self.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)
        boxes, scores = self.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

        losses = {}
        if self.training:
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, targets)
            regression_targets = self.box_coder.encode(matched_gt_boxes, anchors)
            loss_objectness, loss_rpn_box_reg = self.compute_loss(
                objectness, pred_bbox_deltas, labels, regression_targets
            )

            losses = {
                "loss_objectness": loss_objectness,
                "loss_rpn_box_reg": loss_rpn_box_reg,
                "loss_sem_consistency": loss_consistency,
                "loss_align_rpn": loss_align_rpn,
            }

        return boxes, losses

class GradientReverseFunction(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def grad_reverse(x, alpha=1.0):
    return GradientReverseFunction.apply(x, alpha)

class TextGuidedFeaturePurifier(nn.Module):
    """
    Instance-Level Prior Disentanglement
    Enhanced text-guided feature purification with:
    1. Channel-Independent Logical Gate
    2. Visual-only mask predictor (Student) for inference-time feature purification
    """

    def __init__(
        self,
        in_channels=1024,
        embed_dim=2048,
        num_ids=5532,
        lambda_confuse=0.1,
        lambda_cons=1.0,
        tau=0.07,
        num_heads=8,
        num_mask_layers=2
    ):
        super().__init__()

        self.lambda_confuse = lambda_confuse
        self.lambda_cons = lambda_cons
        self.tau = tau
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.visual_projector = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.text_projector = nn.Sequential(
            nn.Linear(in_channels, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        self.id_classifier = nn.Linear(embed_dim, num_ids, bias=False)

        self.w_v = nn.Parameter(torch.ones(embed_dim) * 0.5)
        self.w_t = nn.Parameter(torch.ones(embed_dim) * 0.5)
        self.b = nn.Parameter(torch.zeros(embed_dim))

        self.visual_mask_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        nn.init.orthogonal_(self.visual_projector[0].weight)

        self.logit_scale = nn.Parameter(torch.ones([]) * 2.6592)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _compute_logical_gate_mask(self, v_emb, t_emb):
        """
        Channel-Independent Logical Gate
        """
        gate_input = self.w_v * v_emb + self.w_t * t_emb
        mask = torch.sigmoid(gate_input)

        return mask, gate_input

    def forward(self, roi_feat, text_feat=None, labels=None):
        """
        Args:
            roi_feat: ROI visual features [B, D]
            text_feat: text features [B, D] (required during training)
            labels: ID labels [B]
        """
        v_emb = F.normalize(self.visual_projector(roi_feat), dim=1)

        if not self.training or text_feat is None:
            mask = self.visual_mask_predictor(v_emb)
            v_purified = v_emb * mask
            v_purified = F.normalize(v_purified, dim=1)

            return v_purified

        t_emb = F.normalize(self.text_projector(text_feat), dim=1)

        mask, tea_logits = self._compute_logical_gate_mask(v_emb, t_emb)

        v_keep = v_emb * mask
        v_discard = v_emb * (1 - mask)

        sim_matrix = v_emb @ t_emb.t() * self.logit_scale.exp()
        align_labels = torch.arange(v_emb.size(0), device=v_emb.device)
        loss_align = (F.cross_entropy(sim_matrix, align_labels) +
                     F.cross_entropy(sim_matrix.t(), align_labels)) / 2

        loss_anchor = (1 - F.cosine_similarity(F.normalize(v_keep, dim=1), t_emb, dim=1).mean()) * 5

        v_keep_norm = F.normalize(v_keep, dim=1)
        v_discard_norm = F.normalize(v_discard, dim=1)

        logits_keep = self.id_classifier(v_keep_norm)
        logits_discard = self.id_classifier(v_discard_norm)
        loss_id = F.cross_entropy(logits_keep, labels)

        prob_keep = F.softmax(logits_keep, dim=1)
        prob_discard = F.softmax(logits_discard, dim=1)

        batch_size = v_emb.size(0)
        p_y_keep = prob_keep[torch.arange(batch_size), labels]
        p_y_discard = prob_discard[torch.arange(batch_size), labels]

        diff = torch.clamp(p_y_keep - p_y_discard, min=1e-7)
        loss_confuse = -torch.log(diff).mean()

        stu_logits = self.visual_mask_predictor(v_emb.detach())
        loss_distill = F.mse_loss(stu_logits, tea_logits.detach()) * 10.0

        final_feat = F.normalize(v_keep, dim=1)

        return final_feat, loss_anchor, loss_confuse, loss_align, loss_id, loss_distill


class SafeRegionProposalNetwork(RegionProposalNetwork):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, objectness, pred_bbox_deltas,
            labels, regression_targets):
        """
        Args:
            objectness (Tensor)
            pred_bbox_deltas (Tensor)
            labels (List[Tensor])
            regression_targets (List[Tensor])
        Returns:
            objectness_loss (Tensor)
            box_loss (Tensor)
        """

        sampled_pos_inds, sampled_neg_inds = self.fg_bg_sampler(labels)
        sampled_pos_inds = torch.where(torch.cat(sampled_pos_inds, dim=0))[0]
        sampled_neg_inds = torch.where(torch.cat(sampled_neg_inds, dim=0))[0]

        sampled_inds = torch.cat([sampled_pos_inds, sampled_neg_inds], dim=0)

        objectness = objectness.flatten()

        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)

        box_loss = (
            F.smooth_l1_loss(
                pred_bbox_deltas[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1 / 9,
                reduction="none",
            )
            / (sampled_inds.numel())
        ).sum()

        _objectness_loss = F.binary_cross_entropy_with_logits(objectness[sampled_inds], labels[sampled_inds], reduction='none')
        objectness_loss = (_objectness_loss / _objectness_loss.size(0)).sum()

        return objectness_loss, box_loss

class MGSP(nn.Module):
    def __init__(self, cfg):
        super(MGSP, self).__init__()

        clip_model = open_clip.create_model('RN50', pretrained='openai', force_quick_gelu=True)
        backbone, box_head = build_clip_resnet(clip_model)

        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )

        rpn = PrototypeGuidedRPN(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,

            backbone_out_dim=1024,
            num_prototypes=8
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=(14, 14), sampling_ratio=4
        )
        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
        roi_heads = CascadedROIHeads(
            cfg=cfg,
            faster_rcnn_predictor=faster_rcnn_predictor,
            box_head_2nd=deepcopy(box_head),
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
            clip_model=clip_model,
        )

        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform
        self.eval_feat = cfg.EVAL_FEATURE

        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_rcnn_reg_1st = cfg.SOLVER.LW_RCNN_REG_1ST
        self.lw_rcnn_cls_1st = cfg.SOLVER.LW_RCNN_CLS_1ST
        self.lw_rcnn_reg_2nd = cfg.SOLVER.LW_RCNN_REG_2ND
        self.lw_rcnn_cls_2nd = cfg.SOLVER.LW_RCNN_CLS_2ND
        self.lw_rcnn_reid_2nd = cfg.SOLVER.LW_RCNN_REID_2ND

    def inference(self, images, targets=None, query_img_as_gallery=False):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features_2nd = self.roi_heads.box_head_2nd(box_features)
            embeddings_2nd, _ = self.roi_heads.embedding_head_2nd(box_features_2nd)
            if self.eval_feat == 'stage2':
                embeddings = embeddings_2nd
            else:
                raise Exception("Unknown evaluation feature name")
            return embeddings.split(1, 0)
        else:
            boxes, _ = self.rpn(images, features, targets)
            detections = self.roi_heads(features, boxes, images.image_sizes, targets, query_img_as_gallery)[0]
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections

    def forward(self, images, targets=None, epoch=0, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        boxes, rpn_losses = self.rpn(images, features, targets, epoch)
        _, rcnn_losses = self.roi_heads(features, boxes, images.image_sizes, targets)

        rpn_losses["loss_rpn_reg"] = rpn_losses.pop("loss_rpn_box_reg")
        rpn_losses["loss_rpn_cls"] = rpn_losses.pop("loss_objectness")
        rpn_losses["loss_sem_consistency"] = rpn_losses.pop("loss_sem_consistency")
        rpn_losses["loss_align_rpn"] = rpn_losses.pop("loss_align_rpn")

        losses = {}
        losses.update(rcnn_losses)
        losses.update(rpn_losses)

        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_rcnn_reg_1st"] *= self.lw_rcnn_reg_1st
        losses["loss_rcnn_cls_1st"] *= self.lw_rcnn_cls_1st
        losses["loss_rcnn_reg_2nd"] *= self.lw_rcnn_reg_2nd
        losses["loss_rcnn_cls_2nd"] *= self.lw_rcnn_cls_2nd
        losses["loss_rcnn_reid_2nd"] *= self.lw_rcnn_reid_2nd

        return losses

class CascadedROIHeads(RoIHeads):
    def __init__(
        self,
        cfg,
        faster_rcnn_predictor,
        box_head_2nd,
        clip_model,
        *args,
        **kwargs
    ):
        super(CascadedROIHeads, self).__init__(*args, **kwargs)

        self.use_diff_thresh=cfg.MODEL.ROI_HEAD.USE_DIFF_THRESH
        self.nms_thresh_1st = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_1ST
        self.nms_thresh_2nd = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND
        self.fg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN
        self.bg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN
        self.fg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND
        self.bg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND

        self.box_predictor_1st = faster_rcnn_predictor
        self.box_predictor_2nd = self.box_predictor

        self.box_head_1st = self.box_head
        self.box_head_2nd = box_head_2nd

        self.use_feature_mask = cfg.MODEL.USE_FEATURE_MASK
        self.feature_mask_size = cfg.MODEL.FEATURE_MASK_SIZE

        embedding_dim = cfg.MODEL.EMBEDDING_DIM
        self.embedding_head_2nd = NormAwareEmbedding(featmap_names=["before_trans", "after_trans"], in_channels=[1024, 2048], dim=embedding_dim)

        num_pids = cfg.MODEL.LOSS.LUT_SIZE
        num_cq_size = cfg.MODEL.LOSS.CQ_SIZE
        oim_momentum = cfg.MODEL.LOSS.OIM_MOMENTUM
        oim_scalar = cfg.MODEL.LOSS.OIM_SCALAR
        self.reid_loss_2nd = OIMLossSafe(embedding_dim, num_pids, num_cq_size, oim_momentum, oim_scalar)

        self.postprocess_proposals = self.postprocess_detections

        self.eval_feat = cfg.EVAL_FEATURE

        self.logit_scale = clip_model.logit_scale

        self.purifier = TextGuidedFeaturePurifier()

    def normalize(self, logit):
        mean = logit.mean(dim=-1, keepdim=True)
        stdv = logit.std(dim=-1, keepdim=True)
        return (logit-mean) / (1e-7 + stdv)

    def _get_gt_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
        return mask

    def _get_other_mask(self, logits, target):
        target = target.reshape(-1)
        mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
        return mask

    def cat_mask(self, t, mask1, mask2):
        t1 = (t * mask1).sum(dim=1, keepdims=True)
        t2 = (t * mask2).sum(1, keepdims=True)
        rt = torch.cat([t1, t2], dim=1)
        return rt

    def box_head_purifier(self, box_features, pid_labels=None, instances_id=None, targets=None, theta=0.2, lambda_weight=0.2):

        logit_scale = self.logit_scale.exp()

        box_features = self.box_head_2nd(box_features)
        if pid_labels is None:
            img_features = self.embedding_head_2nd(box_features)[0]
            clean_feat = self.purifier(img_features)
            box_features['after_trans'] = clean_feat
            return box_features, None, None, None, None, None

        target = torch.cat(pid_labels)
        keep = (target > 0) & (target < 483)
        valid_target = target[keep]
        valid_instance = torch.cat(instances_id)[keep]

        gt_labels = torch.cat([target['labels'] for target in targets],dim=0)
        gt_instances_id = torch.cat([target['instances_id'] for target in targets],dim=0)
        text_features = torch.cat([target['text_features'] for target in targets],dim=0)

        img_features = self.embedding_head_2nd(box_features)[0][keep]

        inst2idx = {ins.item(): i for i, ins in enumerate(gt_instances_id)}

        roi_text_features = text_features[
            torch.tensor([inst2idx[i.item()] for i in valid_instance],
                        device=img_features.device)
        ]

        assert roi_text_features.shape == img_features.shape

        clean_feat, cons_loss, confuse_loss, align_loss, id_loss, distill_loss = self.purifier(img_features, roi_text_features, valid_target)

        reid_feat = box_features['after_trans']

        reid_feat_new = reid_feat.clone()

        clean_feat = clean_feat.to(dtype=reid_feat_new.dtype, device=reid_feat_new.device)
        reid_feat_new[keep] = clean_feat

        box_features['after_trans'] = reid_feat_new

        return box_features, cons_loss, confuse_loss, align_loss, id_loss, distill_loss

    def forward(self, features, boxes, image_shapes, targets=None, query_img_as_gallery=False):
        """
        Arguments:
            features (List[Tensor])
            boxes (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        cws = True
        gt_det_2nd = None
        feats_reid_2nd = None
        targets_reid_2nd = None
        box_pid_labels_2nd = None
        box_instances_id_2nd = None

        if self.training:
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_1st,
                    self.bg_iou_thresh_1st,
                    allow_low_quality_matches=False)

            boxes, _, box_pid_labels_1st, box_reg_targets_1st = self.select_training_samples(
                boxes, targets
            )

        # ------------------- The first stage ------------------ #
        box_features_1st = self.box_roi_pool(features, boxes, image_shapes)
        box_features_1st = self.box_head_1st(box_features_1st)
        box_cls_scores_1st, box_regs_1st = self.box_predictor_1st(box_features_1st["after_trans"])

        if self.training:
            boxes = self.get_boxes(box_regs_1st, boxes, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            if self.use_diff_thresh:
                self.proposal_matcher = det_utils.Matcher(
                    self.fg_iou_thresh_2nd,
                    self.bg_iou_thresh_2nd,
                    allow_low_quality_matches=False)
            boxes, _, box_pid_labels_2nd, box_reg_targets_2nd, box_instances_id_2nd = self.select_training_samples_instance(boxes, targets)

        else:
            orig_thresh = self.nms_thresh
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )

        if not self.training and query_img_as_gallery:
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.box_head_2nd(gt_box_features)
            embeddings, _ = self.embedding_head_2nd(gt_box_features)
            gt_det_2nd = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = gt_det_2nd["boxes"] if gt_det_2nd else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det_2nd else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det_2nd else torch.zeros(0)
            if self.eval_feat == 'stage2':
                embeddings = gt_det_2nd["embeddings"] if gt_det_2nd else torch.zeros(0, 256)
            else:
                raise Exception("Unknown evaluation feature name")
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)

        if not self.training:
            box_features_2nd = self.box_head_2nd(box_features)
            box_embeddings_2nd, _ = self.embedding_head_2nd(box_features_2nd)

        box_features, consis_loss, noise_loss, align_loss, id_loss, distill_loss = self.box_head_purifier(box_features, box_pid_labels_2nd, box_instances_id_2nd, targets)

        box_regs_2nd = self.box_predictor_2nd(box_features["after_trans"])
        box_embeddings_2nd, box_cls_scores_2nd = self.embedding_head_2nd(box_features)

        if box_cls_scores_2nd.dim() == 0:
            box_cls_scores_2nd = box_cls_scores_2nd.unsqueeze(0)

        result, losses = [], {}
        if self.training:
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            box_labels_2nd = [y.clamp(0, 1) for y in box_pid_labels_2nd]
            losses = detection_losses(
                box_cls_scores_1st,
                box_regs_1st,
                box_labels_1st,
                box_reg_targets_1st,
                box_cls_scores_2nd,
                box_regs_2nd,
                box_labels_2nd,
                box_reg_targets_2nd,
            )

            loss_rcnn_reid_2nd, loss_triplet_2nd = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels_2nd, box_instances_id_2nd)

            losses.update(loss_rcnn_reid_2nd=loss_rcnn_reid_2nd)
            losses.update(loss_triplet_2nd=loss_triplet_2nd)

            losses.update(consis_loss=consis_loss)
            losses.update(noise_loss=noise_loss)
            losses.update(align_loss=align_loss)
            losses.update(id_loss=id_loss)
            losses.update(distill_loss=distill_loss)

            for k, v in losses.items():
                if v is None:
                    print(f"[Warning] loss {k} is None")
            has_nan = any([torch.isnan(v) for v in losses.values()])
            if has_nan:
                print(" Detected NaN in losses! Full loss status:")
                for name, value in losses.items():
                    if torch.isnan(value):
                        print(f" Loss '{name}' is NaN")
                    else:
                        print(f"Loss '{name}' = {value.item():.6f}")
        else:
            if self.eval_feat == 'stage2':
                boxes, scores, embeddings_2nd, labels = self.postprocess_boxes(
                    box_cls_scores_2nd,
                    box_regs_2nd,
                    box_embeddings_2nd,
                    boxes,
                    image_shapes,
                    fcs=scores,
                    gt_det=gt_det_2nd,
                    cws=cws,
                )
                self.nms_thresh = orig_thresh

            num_images = len(boxes)
            for i in range(num_images):
                if self.eval_feat == 'stage2':
                    embeddings = embeddings_2nd[i]
                else:
                    raise Exception("Unknown evaluation feature name")
                result.append(
                    dict(
                        boxes=boxes[i],
                        labels=labels[i],
                        scores=scores[i],
                        embeddings=embeddings
                    )
                )

        return result, losses

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if fcs is not None:
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            embeddings = embeddings * pred_scores.view(-1, 1)

        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            labels = torch.ones(scores.size(0), device=device)

            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head_2nd.dim)

            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels

    def select_training_samples_instance(
        self,
        proposals,
        targets,
    ):
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_instances_id = [t["instances_id"] for t in targets]

        proposals = self.add_gt_proposals(proposals, gt_boxes)

        matched_idxs, labels, instances_id = self.assign_targets_to_proposals_instance(proposals, gt_boxes, gt_labels, gt_instances_id)
        sampled_inds = self.subsample(labels)
        matched_gt_boxes = []
        num_images = len(proposals)
        for img_id in range(num_images):
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            instances_id[img_id] = instances_id[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])

        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, instances_id

    def assign_targets_to_proposals_instance(self, proposals, gt_boxes, gt_labels, gt_instance_ids):
        matched_idxs = []
        labels = []
        instance_ids = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_instance_in_image in zip(proposals, gt_boxes, gt_labels, gt_instance_ids):

            if gt_boxes_in_image.numel() == 0:
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                instance_in_image = gt_instance_in_image[clamped_matched_idxs_in_image]
                instance_in_image = instance_in_image.to(dtype=torch.int64)

                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                instance_in_image[bg_inds] = 0

                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1
                instance_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            instance_ids.append(instance_in_image)
        return matched_idxs, labels, instance_ids

    @staticmethod
    def grid_wise_label_gen(matched_gt_boxes, proposals, size=(7, 7)):
        cat_gt = torch.cat(matched_gt_boxes, dim=0)
        cat_p = torch.cat(proposals, dim=0)
        num_proposals = cat_gt.size(0)
        grid_wise_labels = torch.zeros(
            num_proposals, 1, *size).to(proposals[0].device)
        for i in range(num_proposals):
            width = (cat_p[i][2] - cat_p[i][0]).ceil().long().item()
            height = (cat_p[i][3] - cat_p[i][1]).ceil().long().item()
            if not (width > 0 and height > 0):
                continue
            tmp = torch.zeros(1, 1, height, width)
            x1 = (torch.max(cat_p[i][0], cat_gt[i][0]) -
                  cat_p[i][0]).floor().long().item()
            y1 = (torch.max(cat_p[i][1], cat_gt[i][1]) -
                  cat_p[i][1]).floor().long().item()
            x2 = (torch.min(cat_p[i][2], cat_gt[i][2]) -
                  cat_p[i][0]).ceil().long().item()
            y2 = (torch.min(cat_p[i][3], cat_gt[i][3]) -
                  cat_p[i][1]).ceil().long().item()
            tmp[0, 0, y1:y2 + 1, x1:x2 + 1] = 1.0
            grid_wise_labels[i] = F.interpolate(
                tmp, size=size, mode='bilinear', align_corners=False)
        return grid_wise_labels


class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256):
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        if len(featmaps) == 1:
            k, v = list(featmaps.items())[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                if k not in self.featmap_names:
                    continue
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp

class PixelWiseNormAwareEmbeddingProj(NormAwareEmbedding):
    '''
    Current Version:
        1. conv kernel shared by all locations
        2. average pool instead of resize/concat, so the output dim = embedding dim
    TODO: Version 2: don't share conv kernels.
    '''

    def __init__(self, *args, **kwargs):
        super(PixelWiseNormAwareEmbeddingProj, self).__init__(*args, **kwargs)
        self.rescaler = nn.BatchNorm2d(1, affine=True)
        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(
                nn.Conv2d(in_chennel, indv_dim, kernel_size=1),
                nn.BatchNorm2d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

    def forward(self, featmaps):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim, gird_size[0], grid_size[1]), L2 normalized embeddings.
            tensor of size (BatchSize, 1, gird_size[0], grid_size[1]) rescaled norm of embeddings, as class_logits.
        '''
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            embeddings = self.projectors[k](v)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.clamp(min=1e-12)
            norms = self.rescaler(norms)
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                outputs.append(
                    self.projectors[k](v)
                )
            outputs[0] = F.interpolate(
                outputs[0], size=outputs[1].shape[2:],
                mode='bilinear', align_corners=False)
            embeddings = torch.cat(outputs, dim=1)
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.clamp(min=1e-12)
            norms = self.rescaler(norms)
            spatial_attention = torch.sigmoid(norms)
            embeddings = F.adaptive_avg_pool2d(
                embeddings * spatial_attention, 1).flatten(start_dim=1)
            if self.training:
                return embeddings, norms
            else:
                return embeddings, F.adaptive_avg_pool2d(spatial_attention, 1).flatten(start_dim=1)

    @property
    def rescaler_weight(self):
        return self.rescaler.weight.item()

class ResEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, in_channel=1024, dim=1024):
        super(ResEmbedding, self).__init__()
        self.dim = dim

        self.res_projectors = nn.Sequential(nn.Linear(in_channel, dim), nn.BatchNorm1d(dim))

    def forward(self, x):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        embeddings = self.res_projectors(x)
        norms = embeddings.norm(2, 1, keepdim=True)
        embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)

        return embeddings


class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_max_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def detection_losses(
    box_cls_scores_1st,
    box_regs_1st,
    box_labels_1st,
    box_reg_targets_1st,
    box_cls_scores_2nd,
    box_regs_2nd,
    box_labels_2nd,
    box_reg_targets_2nd,
):
    # --------------------- The first stage -------------------- #
    box_labels_1st = torch.cat(box_labels_1st, dim=0)
    box_reg_targets_1st = torch.cat(box_reg_targets_1st, dim=0)
    _loss_rcnn_cls_1st = F.cross_entropy(box_cls_scores_1st, box_labels_1st, reduction='none')
    loss_rcnn_cls_1st = (_loss_rcnn_cls_1st/ _loss_rcnn_cls_1st.size(0)).sum()
    sampled_pos_inds_subset = torch.nonzero(box_labels_1st > 0).squeeze(1)
    labels_pos = box_labels_1st[sampled_pos_inds_subset]
    N = box_cls_scores_1st.size(0)
    box_regs_1st = box_regs_1st.reshape(N, -1, 4)

    loss_rcnn_reg_1st = F.smooth_l1_loss(
        box_regs_1st[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_1st[sampled_pos_inds_subset],
        reduction='none'
    )
    loss_rcnn_reg_1st = (loss_rcnn_reg_1st / box_labels_1st.numel()).sum()

    # --------------------- The second stage -------------------- #
    box_labels_2nd = torch.cat(box_labels_2nd, dim=0)
    box_reg_targets_2nd = torch.cat(box_reg_targets_2nd, dim=0)
    _loss_rcnn_cls_2nd = F.binary_cross_entropy_with_logits(box_cls_scores_2nd, box_labels_2nd.float(), reduction='none')
    loss_rcnn_cls_2nd = (_loss_rcnn_cls_2nd/_loss_rcnn_cls_2nd.size(0)).sum()

    sampled_pos_inds_subset = torch.nonzero(box_labels_2nd > 0).squeeze(1)
    labels_pos = box_labels_2nd[sampled_pos_inds_subset]
    N = box_cls_scores_2nd.size(0)
    box_regs_2nd = box_regs_2nd.reshape(N, -1, 4)

    loss_rcnn_reg_2nd = F.smooth_l1_loss(
        box_regs_2nd[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_2nd[sampled_pos_inds_subset],
        reduction='none',
    )
    loss_rcnn_reg_2nd = (loss_rcnn_reg_2nd / box_labels_2nd.numel()).sum()

    return dict(
        loss_rcnn_cls_1st=loss_rcnn_cls_1st,
        loss_rcnn_reg_1st=loss_rcnn_reg_1st,
        loss_rcnn_cls_2nd=loss_rcnn_cls_2nd,
        loss_rcnn_reg_2nd=loss_rcnn_reg_2nd,
    )


def spatial_norm_detection_losses(
    box_cls_scores_1st,
    box_regs_1st,
    box_labels_1st,
    box_reg_targets_1st,
    box_cls_scores_2nd,
    box_regs_2nd,
    box_labels_2nd,
    box_reg_targets_2nd,
    spatial_labels_2nd,
    focal=False,
    alpha_d=0.25,
    gamma_d=2.0
):
    # --------------------- The first stage -------------------- #
    box_labels_1st = torch.cat(box_labels_1st, dim=0)
    box_reg_targets_1st = torch.cat(box_reg_targets_1st, dim=0)
    _loss_rcnn_cls_1st = F.cross_entropy(box_cls_scores_1st, box_labels_1st, reduction='none')
    loss_rcnn_cls_1st = (_loss_rcnn_cls_1st/ _loss_rcnn_cls_1st.size(0)).sum()
    sampled_pos_inds_subset = torch.nonzero(box_labels_1st > 0).squeeze(1)
    labels_pos = box_labels_1st[sampled_pos_inds_subset]
    N = box_cls_scores_1st.size(0)
    box_regs_1st = box_regs_1st.reshape(N, -1, 4)

    loss_rcnn_reg_1st = F.smooth_l1_loss(
        box_regs_1st[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_1st[sampled_pos_inds_subset],
        reduction='none'
    )
    loss_rcnn_reg_1st = (loss_rcnn_reg_1st / box_labels_1st.numel()).sum()

    # --------------------- The second stage -------------------- #
    box_labels_2nd = torch.cat(box_labels_2nd, dim=0)
    box_reg_targets_2nd = torch.cat(box_reg_targets_2nd, dim=0)
    _loss_rcnn_cls_2nd = F.binary_cross_entropy_with_logits(box_cls_scores_2nd, spatial_labels_2nd, reduction='none')
    pt = torch.exp(-_loss_rcnn_cls_2nd)
    _loss_rcnn_cls_2nd = alpha_d * (1 - pt)**gamma_d * _loss_rcnn_cls_2nd
    loss_rcnn_cls_2nd = _loss_rcnn_cls_2nd.mean()

    sampled_pos_inds_subset = torch.nonzero(box_labels_2nd > 0).squeeze(1)
    labels_pos = box_labels_2nd[sampled_pos_inds_subset]
    N = box_cls_scores_2nd.size(0)
    box_regs_2nd = box_regs_2nd.reshape(N, -1, 4)

    loss_rcnn_reg_2nd = F.smooth_l1_loss(
        box_regs_2nd[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_2nd[sampled_pos_inds_subset],
        reduction='none',
    )
    loss_rcnn_reg_2nd = (loss_rcnn_reg_2nd / box_labels_2nd.numel()).sum()

    return dict(
        loss_rcnn_cls_1st=loss_rcnn_cls_1st,
        loss_rcnn_reg_1st=loss_rcnn_reg_1st,
        loss_rcnn_cls_2nd=loss_rcnn_cls_2nd,
        loss_rcnn_reg_2nd=loss_rcnn_reg_2nd,
    )
