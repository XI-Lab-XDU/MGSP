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
from models.resnet import build_resnet, build_clip_resnet, load_clip_to_cpu, PromptLearner, TextEncoder, build_convext
from models.transformer import TransformerHead

import json
# from .clip import clip
import open_clip


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


class COAT(nn.Module):
    def __init__(self, cfg):
        super(COAT, self).__init__()
        # clip_model = load_clip_to_cpu("RN50", 7, 7)
        # clip_model = open_clip.create_model('convnext_base', pretrained='laion400m_s13b_b51k')
        # clip_model = open_clip.create_model('RN50', pretrained='openai', force_quick_gelu=True)
        # backbone, box_head = build_convext(clip_model)
        # backbone, box_head = build_clip_resnet(clip_model)
        # build_clip_resnet
        backbone, box_head = build_resnet(name="resnet50", pretrained=True)
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
        rpn = SafeRegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )

        # box_head = TransformerHead(
        #     cfg=cfg,
        #     trans_names=cfg.MODEL.TRANSFORMER.NAMES_1ST,
        #     kernel_size=cfg.MODEL.TRANSFORMER.KERNEL_SIZE_1ST,
        #     use_feature_mask=cfg.MODEL.TRANSFORMER.USE_MASK_1ST,
        # )
        # box_head_2nd = TransformerHead(
        #     cfg=cfg,
        #     trans_names=cfg.MODEL.TRANSFORMER.NAMES_2ND,
        #     kernel_size=cfg.MODEL.TRANSFORMER.KERNEL_SIZE_2ND,
        #     use_feature_mask=cfg.MODEL.TRANSFORMER.USE_MASK_2ND,
        # )
        # box_head_3rd = TransformerHead(
        #     cfg=cfg,
        #     trans_names=cfg.MODEL.TRANSFORMER.NAMES_3RD,
        #     kernel_size=cfg.MODEL.TRANSFORMER.KERNEL_SIZE_3RD,
        #     use_feature_mask=cfg.MODEL.TRANSFORMER.USE_MASK_3RD,
        # )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=(14, 14), sampling_ratio=4
        )
        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
        roi_heads = CascadedROIHeads(
            cfg=cfg,
            # Cascade Transformer Head
            faster_rcnn_predictor=faster_rcnn_predictor,
            box_head_2nd=deepcopy(box_head),
            # box_head_3rd=deepcopy(box_head),
            # parent class
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
            # clip
            # clip_model = clip_model,
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

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_rcnn_reg_1st = cfg.SOLVER.LW_RCNN_REG_1ST
        self.lw_rcnn_cls_1st = cfg.SOLVER.LW_RCNN_CLS_1ST
        self.lw_rcnn_reg_2nd = cfg.SOLVER.LW_RCNN_REG_2ND
        self.lw_rcnn_cls_2nd = cfg.SOLVER.LW_RCNN_CLS_2ND
        # self.lw_rcnn_reg_3rd = cfg.SOLVER.LW_RCNN_REG_3RD
        # self.lw_rcnn_cls_3rd = cfg.SOLVER.LW_RCNN_CLS_3RD
        self.lw_rcnn_reid_2nd = cfg.SOLVER.LW_RCNN_REID_2ND
        # self.lw_rcnn_reid_3rd = cfg.SOLVER.LW_RCNN_REID_3RD

    def inference(self, images, targets=None, query_img_as_gallery=False):
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features_2nd = self.roi_heads.box_head_2nd(box_features)
            embeddings_2nd, _ = self.roi_heads.embedding_head_2nd(box_features_2nd)
            # box_features_3rd = self.roi_heads.box_head_3rd(box_features)
            # embeddings_3rd, _ = self.roi_heads.embedding_head_3rd(box_features_3rd)
            if self.eval_feat == 'concat':
                embeddings = embeddings_2nd
            # elif self.eval_feat == 'stage2':
            #     embeddings = embeddings_2nd
            # elif self.eval_feat == 'stage3':
            #     embeddings = embeddings_3rd
            # else:
            #     raise Exception("Unknown evaluation feature name")
            return embeddings.split(1, 0)
        else:
            # gallery
            boxes, _ = self.rpn(images, features, targets)
            detections = self.roi_heads(features, boxes, images.image_sizes, targets, query_img_as_gallery)[0]
            detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
            return detections

    def forward(self, images, targets=None, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        boxes, rpn_losses = self.rpn(images, features, targets)

        _, rcnn_losses = self.roi_heads(features, boxes, images.image_sizes, targets)

        # rename rpn losses to be consistent with detection losses
        rpn_losses["loss_rpn_reg"] = rpn_losses.pop("loss_rpn_box_reg")
        rpn_losses["loss_rpn_cls"] = rpn_losses.pop("loss_objectness")

        losses = {}
        losses.update(rcnn_losses)
        losses.update(rpn_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_rcnn_reg_1st"] *= self.lw_rcnn_reg_1st
        losses["loss_rcnn_cls_1st"] *= self.lw_rcnn_cls_1st
        losses["loss_rcnn_reg_2nd"] *= self.lw_rcnn_reg_2nd
        losses["loss_rcnn_cls_2nd"] *= self.lw_rcnn_cls_2nd
        # losses["loss_rcnn_reg_3rd"] *= self.lw_rcnn_reg_3rd
        # losses["loss_rcnn_cls_3rd"] *= self.lw_rcnn_cls_3rd
        losses["loss_rcnn_reid_2nd"] *= self.lw_rcnn_reid_2nd
        # losses["loss_rcnn_reid_3rd"] *= self.lw_rcnn_reid_3rd

        return losses

class CascadedROIHeads(RoIHeads):
    '''
    https://github.com/pytorch/vision/blob/master/torchvision/models/detection/roi_heads.py
    '''
    def __init__(
        self,
        cfg,
        faster_rcnn_predictor,
        box_head_2nd,
        # box_head_3rd,
        # clip_model,
        *args,
        **kwargs
    ):
        super(CascadedROIHeads, self).__init__(*args, **kwargs)

        # ROI head
        self.use_diff_thresh=cfg.MODEL.ROI_HEAD.USE_DIFF_THRESH
        self.nms_thresh_1st = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_1ST
        self.nms_thresh_2nd = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_2ND
        # self.nms_thresh_3rd = cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST_3RD
        self.fg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN
        self.bg_iou_thresh_1st = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN
        self.fg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN_2ND
        self.bg_iou_thresh_2nd = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_2ND
        # self.fg_iou_thresh_3rd = cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN_3RD
        # self.bg_iou_thresh_3rd = cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN_3RD

        # Regression head
        self.box_predictor_1st = faster_rcnn_predictor
        self.box_predictor_2nd = self.box_predictor
        # self.box_predictor_3rd = deepcopy(self.box_predictor)

        # Transformer head
        self.box_head_1st = self.box_head
        self.box_head_2nd = box_head_2nd
        # self.box_head_3rd = box_head_3rd

        # feature mask
        self.use_feature_mask = cfg.MODEL.USE_FEATURE_MASK
        self.feature_mask_size = cfg.MODEL.FEATURE_MASK_SIZE

        # Feature embedding
        embedding_dim = cfg.MODEL.EMBEDDING_DIM
        self.embedding_head_2nd = NormAwareEmbedding(featmap_names=["before_trans", "after_trans"], in_channels=[1024, 2048], dim=embedding_dim)
        # self.embedding_head_3rd = deepcopy(self.embedding_head_2nd)
        
        # self.softmax_criterion_s2 = SoftmaxLoss(cfg)
        # self.softmax_criterion_s3 = SoftmaxLoss(cfg)
        
        # self.res_mlp_2nd = ResEmbedding(1024, 1024)
        # self.res_mlp_3rd = ResEmbedding(1024, 1024)

        # OIM
        num_pids = cfg.MODEL.LOSS.LUT_SIZE
        num_cq_size = cfg.MODEL.LOSS.CQ_SIZE
        oim_momentum = cfg.MODEL.LOSS.OIM_MOMENTUM
        oim_scalar = cfg.MODEL.LOSS.OIM_SCALAR
        self.reid_loss_2nd = OIMLossSafe(embedding_dim, num_pids, num_cq_size, oim_momentum, oim_scalar)
        # self.reid_loss_3rd = deepcopy(self.reid_loss_2nd)
        # self.reid_loss_2nd_res = deepcopy(self.reid_loss_2nd)
        # self.reid_loss_3rd_res = deepcopy(self.reid_loss_2nd)

        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections

        # evaluation        
        self.eval_feat = cfg.EVAL_FEATURE
        
        # clip
        # self.prompt_learner_2nd = PromptLearner(clip_model)
        # self.text_meta_net_2nd = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(1024, 512)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(512, 512))
        # ]))
        # # self.prompt_learner_3rd = PromptLearner(clip_model)
        # self.text_meta_net_3rd = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(1024, 512)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(512, 512))
        # ]))
        # self.text_encoder = TextEncoder(clip_model)
        # # self.tokenized_prompts_2nd = self.prompt_learner_2nd.tokenized_prompts
        # # self.tokenized_prompts_3rd = self.prompt_learner_3rd.tokenized_prompts
        # self.logit_scale = clip_model.logit_scale
        # with open("/data/zhy/COAT/data/PRW/image2test_qwen2.json", 'r', encoding='utf-8') as file:
        #     data = json.load(file)
        # self.img2text = data
        
    def normalize(self, logit):
        mean = logit.mean(dim=-1, keepdim=True)
        stdv = logit.std(dim=-1, keepdim=True)
        return (logit-mean) / (1e-7 + stdv)
      
    def supconloss(self, text_features, image_features, t_label, i_targets):
        batch_size = text_features.shape[0]
        batch_size_N = image_features.shape[0]
        mask = torch.eq(t_label.unsqueeze(1).expand(batch_size, batch_size_N), \
                    i_targets.unsqueeze(0).expand(batch_size,batch_size_N)).float()
        logits = torch.div(torch.matmul(text_features, image_features.T), 0.07)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - mean_log_prob_pos.mean()  
        return loss

    def calc_contrastive(self, image_features, text_features, sim_targets, alpha=0.1, logit_scale=20):
        with torch.no_grad():
            # sim_i2t_s = logit_scale * image_features_s @ text_features_s_gathered.t()
            # sim_t2i_s = logit_scale * text_features_s @ image_features_s_gathered.t()
            sim_i2t_targets = sim_targets
            # sim_t2i_targets = sim_targets # soft + hard
        sim_i2t = logit_scale * image_features @ text_features.t()
        # sim_t2i = logit_scale * text_features @ image_features.t()
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        # loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_ita = loss_i2t
        return loss_ita
    
    # def cross_entropy_loss(self, logits, target):
    #     return F.cross_entropy(logits, target)
    def kd_loss(self, logits_student, logits_teacher, gt_mask, theta, logit_scale):
        eq_matrix = gt_mask/gt_mask.sum(dim=-1, keepdim=True)
        log_pred_student = F.log_softmax(logits_student * logit_scale, dim=1)
        pred_teacher = F.softmax((logits_teacher) * logit_scale, dim=1) * theta +  eq_matrix * (1 - theta)
        kd_loss = (
            F.kl_div(log_pred_student, pred_teacher, reduction="none")
            / log_pred_student.shape[0]
        ).sum()
        return kd_loss
        
    def dkd_loss(self, logits_student, logits_teacher, gt_mask, theta, logit_scale, alpha, beta):
        # gt_mask = self._get_gt_mask(logits_student, target)
        # other_mask = self._get_other_mask(logits_student, target)
        other_mask = ~gt_mask
        eq_matrix = gt_mask/gt_mask.sum(dim=-1, keepdim=True)
        # pred_student = F.softmax(logits_student * logit_scale, dim=1)
        # pred_teacher = F.softmax(logits_teacher * logit_scale, dim=1) * theta + eq_matrix * (1 - theta)
        # pred_student = self.cat_mask(pred_student, gt_mask, other_mask)
        # pred_teacher = self.cat_mask(pred_teacher, gt_mask, other_mask)
        # log_pred_student = torch.log(pred_student)
        # tckd_loss = (
        #     F.kl_div(log_pred_student, pred_teacher, reduction="none")
        #     / log_pred_student.shape[0]
        # ).sum()
        pred_teacher_part2 = F.softmax(
            logits_teacher * logit_scale, dim=1
        ) * theta + eq_matrix * (1 - theta)
        log_pred_student_part2 = F.log_softmax(
            logits_student * logit_scale, dim=1
        )
        _kd_loss = (
            F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="none")
            / log_pred_student_part2.shape[0]
        )
        
        tckd_loss = (_kd_loss * gt_mask).sum()
        nckd_loss = (_kd_loss * other_mask).sum()
        return tckd_loss, nckd_loss

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
              
    def box_head_2nd_text(self, box_features, pid_labels=None, instances_id=None, targets=None, theta=0.3, lambda_weight=0.2):
        
        logit_scale = self.logit_scale.exp()
        
        box_features = self.box_head_2nd(box_features)
        if pid_labels is None:
            return box_features, None, None
          
        target = torch.cat(pid_labels)
        keep = (target > 0)
        valid_target = target[keep]
        valid_instance = torch.cat(instances_id)[keep]
        
        gt_labels = torch.cat([target['labels'] for target in targets],dim=0)
        gt_instances_id = torch.cat([target['instances_id'] for target in targets],dim=0)
        image_features_fr = torch.cat([target['image_features'] for target in targets],dim=0)
        text_features_fr = torch.cat([target['text_features'] for target in targets],dim=0)
        
        # indices_5554 = torch.where(valid_target == 5554)[0]
        # for i, idx in enumerate(indices_5554):
        #     valid_target[idx] = 5554 + i

        # text_caption = [self.img2text[str(i.item())] for i in valid_instance]
        
        img_features = self.embedding_head_2nd(box_features)[0][keep]
        # img_features = box_features["after_trans"][keep]
        frozen_img_featuers = torch.zeros_like(img_features)
        frozen_text_featuers = torch.zeros_like(img_features)
        for i, ins in enumerate(gt_instances_id):
            mask = valid_instance == ins
            frozen_img_featuers[mask] =  image_features_fr[i]
            frozen_text_featuers[mask] =  text_features_fr[i]
        # img_protype_features = self.reid_loss_2nd.lut[valid_target]
        # prompts = self.prompt_learner_2nd(valid_target)
        # tokenized_prompts = clip.tokenize(text_caption, truncate=True).to(valid_instance.device)
        # # prompt = self.tokenized_prompts(tokenized_prompts).to(box_features.device)
        # text_features = self.text_encoder(tokenized_prompts)
        # text_features = self.text_meta_net_2nd(text_features)
        # text_features = text_features/ text_features.norm(dim=-1, keepdim=True)
        
        # img_features_detach = img_features.detach()
        # logits = img_features_detach @ text_features.t()
        # loss_text = F.cross_entropy(logits, valid_target)
        # with torch.no_grad():
        # eq_matrix = (valid_target.unsqueeze(1) == valid_target.unsqueeze(0))
        # eq_matrix = eq_matrix/eq_matrix.sum(dim=-1, keepdim=True)
        
        # S_TI = frozen_img_featuers @ frozen_img_featuers.t() * 0.5 + eq_matrix * 0.5
        td = frozen_img_featuers.unsqueeze(0) - frozen_img_featuers.unsqueeze(1)
        norm_td = F.normalize(td, 2, dim=2, eps=1e-6)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        # td_text = frozen_text_featuers.unsqueeze(0) - frozen_text_featuers.unsqueeze(1)
        # norm_td_text = F.normalize(td_text, 2, dim=2, eps=1e-6)
        # td_text_angle = torch.bmm(norm_td_text, norm_td_text.transpose(1, 2)).view(-1)


        sd = img_features.unsqueeze(0) - img_features.unsqueeze(1)
        norm_sd = F.normalize(sd, 2, dim=2, eps=1e-6)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        rkd_loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        # print(f"t_angle mean: {t_angle.mean().item()}, std: {t_angle.std().item()}")
        # print(f"s_angle mean: {s_angle.mean().item()}, std: {s_angle.std().item()}")
        # S_SI = img_features @ img_features.t()
        # B = S_TI.shape[0]
        # S_SI = img_features @ img_features.t()
        
        # rkd_loss = torch.norm(S_TI - S_SI, p='fro')**2/(B**2)
            # weight = 1.0 + lambda_weight * S
        
        
        logits_img = (img_features @ text_features_fr.t()) * logit_scale
        logits_img_frozen = (frozen_img_featuers @ text_features_fr.t()) * logit_scale
        log_p = F.log_softmax(logits_img, dim=-1) # (N * M)
        
        eq_matrix = (valid_target.unsqueeze(1) == gt_labels.unsqueeze(0))
        eq_matrix = eq_matrix/eq_matrix.sum(dim=-1, keepdim=True)
        q = F.softmax(logits_img_frozen, dim=-1) * theta + eq_matrix * (1 - theta) # (N * M)
        _kl_loss = F.kl_div(log_p, q, reduction='none') # (N * M)
        kl_loss = (_kl_loss/_kl_loss.size(0)).sum()
        # tckd_loss, nckd_loss = self.dkd_loss(logits_img, logits_img_frozen, eq_matrix, theta, logit_scale, 1, 1)
        
        # sim_target = torch.eq(valid_target.unsqueeze(1), valid_target.unsqueeze(0))
        # sim_target = sim_target/sim_target.sum(dim=1, keepdim=True)

        # loss_ita = self.calc_contrastive(img_features, text_features, sim_target, logit_scale=logit_scale)
        # t2iloss = self.supconloss(text_features, img_features, valid_target, valid_target)
        # i2tloss = self.supconloss(img_features, img_features, valid_target, valid_target)
        # res_img_feature = img_features-frozen_text_featuers
        # res_loss = self.supconloss(res_img_feature, res_img_feature, valid_target, valid_target)\
        # _image_scl_loss = F.l1_loss(img_features, frozen_img_featuers, reduction="none")
        # image_scl_loss = (_image_scl_loss /( _image_scl_loss.size(0) * _image_scl_loss.size(1))).sum()
        # F.cosine_embedding_loss
        return box_features, kl_loss, rkd_loss

    def box_head_3rd_text(self, box_features, pid_labels=None, instances_id=None, targets=None, theta=0.3, lambda_weight=0.1):
        logit_scale = self.logit_scale.exp()
        
        box_features = self.box_head_3rd(box_features)
        if pid_labels is None:
            return box_features, None, None
        
        target = torch.cat(pid_labels)
        keep = (target > 0)
        valid_target = target[keep]
        valid_instance = torch.cat(instances_id)[keep]
        
        gt_labels = torch.cat([target['labels'] for target in targets],dim=0)
        instances_id = torch.cat([target['instances_id'] for target in targets],dim=0)
        image_features_fr = torch.cat([target['image_features'] for target in targets],dim=0)
        text_features_fr = torch.cat([target['text_features'] for target in targets],dim=0)
        # text_caption = [self.img2text[str(i.item())] for i in valid_instance]
        # valid_target = valid_target[valid_target<5555]
        
        # indices_5554 = torch.where(valid_target == 5554)[0]
        # for i, idx in enumerate(indices_5554):
        #     valid_target[idx] = 5554 + i
        
        # img_features = box_features["after_trans"][keep]
        img_features = self.embedding_head_3rd(box_features)[0][keep]
        frozen_img_featuers = torch.zeros_like(img_features)
        frozen_text_featuers = torch.zeros_like(img_features)
        for i, ins in enumerate(instances_id):
            mask = valid_instance == ins
            frozen_img_featuers[mask] =  image_features_fr[i]
            frozen_text_featuers[mask] =  text_features_fr[i]
        # img_protype_features = self.reid_loss_2nd.lut[valid_target]
        # prompts = self.prompt_learner_2nd(valid_target)
        # tokenized_prompts = clip.tokenize(text_caption, truncate=True).to(valid_instance.device)
        # # prompt = self.tokenized_prompts(tokenized_prompts).to(box_features.device)
        # text_features = self.text_encoder(tokenized_prompts)
        # # img_features = img_features / img_features.norm(dim=-1, keepdim=True)
        # # text_features = text_features/ text_features.norm(dim=-1, keepdim=True)
        # text_features = self.text_meta_net_2nd(text_features)
        # text_features = text_features/ text_features.norm(dim=-1, keepdim=True)
        
        # img_features_detach = img_features.detach()
        # logits = img_features_detach @ text_features.t()
        # loss_text = F.cross_entropy(logits, valid_target)
        
        # with torch.no_grad():
        #     S = frozen_text_featuers @ text_features_fr.t()
        #     weight = 1.0 + lambda_weight * S
        # eq_matrix = (valid_target.unsqueeze(1) == valid_target.unsqueeze(0))
        # eq_matrix = eq_matrix/eq_matrix.sum(dim=-1, keepdim=True)
        
        td = frozen_img_featuers.unsqueeze(0) - frozen_img_featuers.unsqueeze(1)
        norm_td = F.normalize(td, 2, dim=2, eps=1e-6)
        t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        # td_text = frozen_text_featuers.unsqueeze(0) - frozen_text_featuers.unsqueeze(1)
        # norm_td_text = F.normalize(td_text, 2, dim=2, eps=1e-6)
        # td_text_angle = torch.bmm(norm_td_text, norm_td_text.transpose(1, 2)).view(-1)


        sd = img_features.unsqueeze(0) - img_features.unsqueeze(1)
        norm_sd = F.normalize(sd, 2, dim=2, eps=1e-6)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
        rkd_loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
        
        logits_img = (img_features @ text_features_fr.t()) * logit_scale
        logits_img_frozen = (frozen_img_featuers @ text_features_fr.t()) * logit_scale
        log_p = F.log_softmax(logits_img, dim=-1) # (N * M)
        
        eq_matrix = (valid_target.unsqueeze(1) == gt_labels.unsqueeze(0))
        eq_matrix = eq_matrix/eq_matrix.sum(dim=-1, keepdim=True)
        q = F.softmax(logits_img_frozen, dim=-1) * theta + eq_matrix * (1 - theta) # (N * M)
        _kl_loss = F.kl_div(log_p, q, reduction='none') # (N * M)
        kl_loss = (_kl_loss/_kl_loss.size(0)).sum()
        # tckd_loss, nckd_loss = self.dkd_loss(logits_img, logits_img_frozen, eq_matrix, theta, logit_scale, 1, 1)
        
        # sim_target = torch.eq(valid_target.unsqueeze(1), valid_target.unsqueeze(0))
        # sim_target = sim_target/sim_target.sum(dim=1, keepdim=True)

        # loss_ita = self.calc_contrastive(img_features, text_features, sim_target, logit_scale=logit_scale)
        # t2iloss = self.supconloss(text_features, img_features, valid_target, valid_target)
        # i2tloss = self.supconloss(img_features, img_features, valid_target, valid_target)
        # res_img_feature = img_features-frozen_text_featuers
        # res_loss = self.supconloss(res_img_feature, res_img_feature, valid_target, valid_target)\
        # _image_scl_loss = F.l1_loss(img_features, frozen_img_featuers, reduction="none")
        # image_scl_loss = (_image_scl_loss /( _image_scl_loss.size(0) * _image_scl_loss.size(1))).sum()
        # F.cosine_embedding_loss
        return box_features, kl_loss, rkd_loss
        
        

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
        # gt_det_3rd = None
        # feats_reid_2nd = None
        # feats_reid_3rd = None
        # targets_reid_2nd = None
        # targets_reid_3rd = None
        box_pid_labels_2nd = None
        # box_pid_labels_3rd = None
        box_instances_id_2nd = None
        # box_instances_id_3rd= None

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
            # spatial_labels_2nd = self.grid_wise_label_gen(matched_gt_boxes, boxes,
            #                                     size=(7,7))
        
        else:
            orig_thresh = self.nms_thresh # 0.4
            self.nms_thresh = self.nms_thresh_1st
            boxes, scores, _ = self.postprocess_proposals(
                box_cls_scores_1st, box_regs_1st, boxes, image_shapes
            )

        if not self.training and query_img_as_gallery:
            # When regarding the query image as gallery, GT boxes may be excluded
            # from detected boxes. To avoid this, we compulsorily include GT in the
            # detection results. Additionally, CWS should be disabled as the
            # confidences of these people in query image are 1
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
            if self.eval_feat == 'concat':
                embeddings = torch.cat((gt_det_2nd["embeddings"], gt_det_2nd["embeddings"]), dim=1) if gt_det_2nd else torch.zeros(0, 512)
            elif self.eval_feat == 'stage2' or self.eval_feat == 'stage3':
                embeddings = gt_det_2nd["embeddings"] if gt_det_2nd else torch.zeros(0, 256)
            else:
                raise Exception("Unknown evaluation feature name")
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- The second stage -------------------- #
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        # box_features, kd_loss_2nd, rkd_loss_2nd = self.box_head_2nd_text(box_features, box_pid_labels_2nd, box_instances_id_2nd, targets)
        box_features = self.box_head_2nd(box_features)
        box_regs_2nd = self.box_predictor_2nd(box_features["after_trans"])
        box_embeddings_2nd, box_cls_scores_2nd = self.embedding_head_2nd(box_features)

        
        if box_cls_scores_2nd.dim() == 0:
            box_cls_scores_2nd = box_cls_scores_2nd.unsqueeze(0)

        # if self.training:
        #     boxes = self.get_boxes(box_regs_2nd, boxes, image_shapes)
        #     boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
        #     if self.use_diff_thresh:
        #         self.proposal_matcher = det_utils.Matcher(
        #             self.fg_iou_thresh_3rd,
        #             self.bg_iou_thresh_3rd,
        #             allow_low_quality_matches=False)
        #     boxes, _, box_pid_labels_3rd, box_reg_targets_3rd, box_instances_id_3rd = self.select_training_samples_instance(boxes, targets)
        #     # spatial_labels_3rd = self.grid_wise_label_gen(matched_gt_boxes, boxes,
        #     #                                     size=(7,7))
        # else:
        #     self.nms_thresh = self.nms_thresh_2nd
        #     if self.eval_feat != 'stage2':
        #         boxes, scores, _, _ = self.postprocess_boxes(
        #             box_cls_scores_2nd,
        #             box_regs_2nd,
        #             box_embeddings_2nd,
        #             boxes,
        #             image_shapes,
        #             fcs=scores,
        #             gt_det=None,
        #             cws=cws,
        #         )

        # if not self.training and query_img_as_gallery and self.eval_feat != 'stage2':
        #     cws = False
        #     gt_box = [targets[0]["boxes"]]
        #     gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
        #     gt_box_features = self.box_head_3rd(gt_box_features)
        #     embeddings, _ = self.embedding_head_3rd(gt_box_features)
        #     gt_det_3rd = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # # no detection predicted by Faster R-CNN head in test phase
        # if boxes[0].shape[0] == 0 and self.eval_feat != 'stage2':
        #     assert not self.training
        #     boxes = gt_det_3rd["boxes"] if gt_det_3rd else torch.zeros(0, 4)
        #     labels = torch.ones(1).type_as(boxes) if gt_det_3rd else torch.zeros(0)
        #     scores = torch.ones(1).type_as(boxes) if gt_det_3rd else torch.zeros(0)
        #     if self.eval_feat == 'concat':
        #         embeddings = torch.cat((gt_det_2nd["embeddings"], gt_det_3rd["embeddings"]), dim=1) if gt_det_3rd else torch.zeros(0, 512)
        #     elif self.eval_feat == 'stage3':
        #         embeddings = gt_det_2nd["embeddings"] if gt_det_3rd else torch.zeros(0, 256)
        #     else:
        #         raise Exception("Unknown evaluation feature name")
        #     return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # # --------------------- The third stage -------------------- #
        # box_features = self.box_roi_pool(features, boxes, image_shapes)

        # if not self.training:
        #     box_features_2nd = self.box_head_2nd(box_features)
        #     box_embeddings_2nd, _ = self.embedding_head_2nd(box_features_2nd)

        # box_features, kd_loss_3rd, rkd_loss_3rd = self.box_head_3rd_text(box_features, box_pid_labels_3rd, box_instances_id_3rd, targets)
        # # box_features = self.box_head_3rd(box_features)
        # box_regs_3rd = self.box_predictor_3rd(box_features["after_trans"])
        # box_embeddings_3rd, box_cls_scores_3rd = self.embedding_head_3rd(box_features)
        # if box_cls_scores_3rd.dim() == 0:
        #     box_cls_scores_3rd = box_cls_scores_3rd.unsqueeze(0)

        result, losses = [], {}
        if self.training:
            box_labels_1st = [y.clamp(0, 1) for y in box_pid_labels_1st]
            box_labels_2nd = [y.clamp(0, 1) for y in box_pid_labels_2nd]
            # box_labels_3rd = [y.clamp(0, 1) for y in box_pid_labels_3rd]
            losses = detection_losses(
                box_cls_scores_1st,
                box_regs_1st,
                box_labels_1st,
                box_reg_targets_1st,
                box_cls_scores_2nd,
                box_regs_2nd,
                box_labels_2nd,
                box_reg_targets_2nd,
                # box_cls_scores_3rd,
                # box_regs_3rd,
                # box_labels_3rd,
                # box_reg_targets_3rd,
            )

            loss_rcnn_reid_2nd, loss_triplet_2nd = self.reid_loss_2nd(box_embeddings_2nd, box_pid_labels_2nd, box_instances_id_2nd)
            # loss_rcnn_reid_3rd, loss_triplet_3rd = self.reid_loss_3rd(box_embeddings_3rd, box_pid_labels_3rd, box_instances_id_3rd)
            # softmax_loss_2nd = 0.5 * self.softmax_criterion_s2(feats_reid_2nd, targets_reid_2nd)
            # softmax_loss_3rd = 0.5 * self.softmax_criterion_s3(feats_reid_3rd, targets_reid_3rd)
            
            losses.update(loss_rcnn_reid_2nd=loss_rcnn_reid_2nd)
            # losses.update(loss_rcnn_reid_3rd=loss_rcnn_reid_3rd)
            # losses.update(loss_triplet_2nd=loss_triplet_2nd)
            # losses.update(loss_triplet_3rd=loss_triplet_3rd)
            # # losses.update(ita_loss_3rd=ita_loss_3rd*0.1)
            # # losses.update(i2t_loss_3rd=i2t_loss_3rd*0.1)
            # losses.update(kd_loss_2nd=kd_loss_2nd)
            # losses.update(kd_loss_3rd=kd_loss_3rd)
            # losses.update(rkd_loss_2nd=rkd_loss_2nd)
            # losses.update(rkd_loss_3rd=rkd_loss_3rd)
            # losses.update(softmax_loss_2nd=softmax_loss_2nd)
            # losses.update(softmax_loss_3rd=softmax_loss_3rd)
        else:
            # if self.eval_feat == 'stage2':
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
            # else:
            #     self.nms_thresh = self.nms_thresh_3rd
            #     _, _, embeddings_2nd, _ = self.postprocess_boxes(
            #         box_cls_scores_3rd,
            #         box_regs_3rd,
            #         box_embeddings_2nd,
            #         boxes,
            #         image_shapes,
            #         fcs=scores,
            #         gt_det=gt_det_2nd,
            #         cws=cws,
            #     )
            #     boxes, scores, embeddings_3rd, labels = self.postprocess_boxes(
            #         box_cls_scores_3rd,
            #         box_regs_3rd,
            #         box_embeddings_3rd,
            #         boxes,
            #         image_shapes,
            #         fcs=scores,
            #         gt_det=gt_det_3rd,
            #         cws=cws,
            #     )
            #     # set to original thresh after finishing postprocess
            #     self.nms_thresh = orig_thresh

            num_images = len(boxes)
            for i in range(num_images):
                if self.eval_feat == 'concat':
                    embeddings = embeddings_2nd[i]
                # elif self.eval_feat == 'stage2':
                #     embeddings = embeddings_2nd[i]
                # elif self.eval_feat == 'stage3':
                #     embeddings = embeddings_3rd[i]
                # else:
                #     raise Exception("Unknown evaluation feature name")
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
            # remove predictions with the background label
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
            # Fist Classification Score (FCS)
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

        # split boxes and scores per image
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

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()
            embeddings = embeddings.reshape(-1, self.embedding_head_2nd.dim)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
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
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        dtype = proposals[0].dtype
        device = proposals[0].device

        gt_boxes = [t["boxes"].to(dtype) for t in targets]
        gt_labels = [t["labels"] for t in targets]
        gt_instances_id = [t["instances_id"] for t in targets]

        # append ground-truth bboxes to propos
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices for each proposal
        matched_idxs, labels, instances_id = self.assign_targets_to_proposals_instance(proposals, gt_boxes, gt_labels, gt_instances_id)
        # sample a fixed proportion of positive-negative proposals
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
        # type: (List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
        matched_idxs = []
        labels = []
        instance_ids = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_instance_in_image in zip(proposals, gt_boxes, gt_labels, gt_instance_ids):

            if gt_boxes_in_image.numel() == 0:
                # Background image
                device = proposals_in_image.device
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                #  set to self.box_similarity when https://github.com/pytorch/pytorch/issues/27495 lands
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image)
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix)

                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)

                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)
                
                instance_in_image = gt_instance_in_image[clamped_matched_idxs_in_image]
                instance_in_image = instance_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                instance_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
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
                continue  # invalid proposal
            tmp = torch.zeros(1, 1, height, width)  # same 'size' as proposal
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

    # def forward(self, featmaps):
    #     embeddings = featmaps["after_trans"]
    #     norms = embeddings.norm(2, 1, keepdim=True)
    #     embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
    #     norms = self.rescaler(norms).squeeze()
    #     return embeddings, norms
    
    def forward(self, featmaps):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        # assert len(featmaps) == len(self.featmap_names)
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
        # self.featmap_names = featmap_names
        # self.in_channels = in_channels
        self.dim = dim

        self.res_projectors = nn.Sequential(nn.Linear(in_channel, dim), nn.BatchNorm1d(dim))
        # indv_dims = self._split_embedding_dim()
        # for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
        #     proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
        #     init.normal_(proj[0].weight, std=0.01)
        #     init.normal_(proj[1].weight, std=0.01)
        #     init.constant_(proj[0].bias, 0)
        #     init.constant_(proj[1].bias, 0)
        #     self.projectors[ftname] = proj

        # self.rescaler = nn.BatchNorm1d(1, affine=True)

    def forward(self, x):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        # assert len(featmaps) == len(self.featmap_names)
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
    # box_cls_scores_3rd,
    # box_regs_3rd,
    # box_labels_3rd,
    # box_reg_targets_3rd,
):
    # --------------------- The first stage -------------------- #
    box_labels_1st = torch.cat(box_labels_1st, dim=0)
    box_reg_targets_1st = torch.cat(box_reg_targets_1st, dim=0)
    _loss_rcnn_cls_1st = F.cross_entropy(box_cls_scores_1st, box_labels_1st, reduction='none')    
    loss_rcnn_cls_1st = (_loss_rcnn_cls_1st/ _loss_rcnn_cls_1st.size(0)).sum()
    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
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

    # # --------------------- The third stage -------------------- #
    # box_labels_3rd = torch.cat(box_labels_3rd, dim=0)
    # box_reg_targets_3rd = torch.cat(box_reg_targets_3rd, dim=0)
    # _loss_rcnn_cls_3rd = F.binary_cross_entropy_with_logits(box_cls_scores_3rd, box_labels_3rd.float(), reduction='none')
    # loss_rcnn_cls_3rd = (_loss_rcnn_cls_3rd/ _loss_rcnn_cls_3rd.size(0)).sum()
    
    # sampled_pos_inds_subset = torch.nonzero(box_labels_3rd > 0).squeeze(1)
    # labels_pos = box_labels_3rd[sampled_pos_inds_subset]
    # N = box_cls_scores_3rd.size(0)
    # box_regs_3rd = box_regs_3rd.reshape(N, -1, 4)

    # loss_rcnn_reg_3rd = F.smooth_l1_loss(
    #     box_regs_3rd[sampled_pos_inds_subset, labels_pos],
    #     box_reg_targets_3rd[sampled_pos_inds_subset],
    #     reduction='none',
    # )
    # loss_rcnn_reg_3rd = (loss_rcnn_reg_3rd / box_labels_3rd.numel()).sum()

    return dict(
        loss_rcnn_cls_1st=loss_rcnn_cls_1st,
        loss_rcnn_reg_1st=loss_rcnn_reg_1st,
        loss_rcnn_cls_2nd=loss_rcnn_cls_2nd,
        loss_rcnn_reg_2nd=loss_rcnn_reg_2nd,
        # loss_rcnn_cls_3rd=loss_rcnn_cls_3rd,
        # loss_rcnn_reg_3rd=loss_rcnn_reg_3rd,
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
    box_cls_scores_3rd,
    box_regs_3rd,
    box_labels_3rd,
    box_reg_targets_3rd,
    spatial_labels_3rd,
    focal=False, 
    alpha_d=0.25, 
    gamma_d=2.0
):
    # --------------------- The first stage -------------------- #
    box_labels_1st = torch.cat(box_labels_1st, dim=0)
    box_reg_targets_1st = torch.cat(box_reg_targets_1st, dim=0)
    _loss_rcnn_cls_1st = F.cross_entropy(box_cls_scores_1st, box_labels_1st, reduction='none')    
    loss_rcnn_cls_1st = (_loss_rcnn_cls_1st/ _loss_rcnn_cls_1st.size(0)).sum()
    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
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
    # _loss_rcnn_cls_2nd = F.binary_cross_entropy_with_logits(box_cls_scores_2nd, box_labels_2nd.float(), reduction='none')
    # loss_rcnn_cls_2nd = (_loss_rcnn_cls_2nd/_loss_rcnn_cls_2nd.size(0)).sum()
    _loss_rcnn_cls_2nd = F.binary_cross_entropy_with_logits(box_cls_scores_2nd, spatial_labels_2nd, reduction='none')
    pt = torch.exp(-_loss_rcnn_cls_2nd)
    _loss_rcnn_cls_2nd = alpha_d * (1 - pt)**gamma_d * _loss_rcnn_cls_2nd
    loss_rcnn_cls_2nd = _loss_rcnn_cls_2nd.mean()
    # _loss_rcnn_cls_2nd = (_loss_rcnn_cls_2nd/49)
    # loss_rcnn_cls_2nd = (_loss_rcnn_cls_2nd / _loss_rcnn_cls_2nd.size(0)).sum()
    
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

    # --------------------- The third stage -------------------- #
    box_labels_3rd = torch.cat(box_labels_3rd, dim=0)
    box_reg_targets_3rd = torch.cat(box_reg_targets_3rd, dim=0)
    # _loss_rcnn_cls_3rd = F.binary_cross_entropy_with_logits(box_cls_scores_3rd, box_labels_3rd.float(), reduction='none')
    # loss_rcnn_cls_3rd = (_loss_rcnn_cls_3rd/ _loss_rcnn_cls_3rd.size(0)).sum()
    _loss_rcnn_cls_3rd = F.binary_cross_entropy_with_logits(box_cls_scores_3rd, spatial_labels_3rd, reduction='none')
    pt = torch.exp(-_loss_rcnn_cls_3rd)
    _loss_rcnn_cls_3rd = alpha_d * (1 - pt)**gamma_d * _loss_rcnn_cls_3rd
    loss_rcnn_cls_3rd = _loss_rcnn_cls_3rd.mean()
    
    sampled_pos_inds_subset = torch.nonzero(box_labels_3rd > 0).squeeze(1)
    labels_pos = box_labels_3rd[sampled_pos_inds_subset]
    N = box_cls_scores_3rd.size(0)
    box_regs_3rd = box_regs_3rd.reshape(N, -1, 4)

    loss_rcnn_reg_3rd = F.smooth_l1_loss(
        box_regs_3rd[sampled_pos_inds_subset, labels_pos],
        box_reg_targets_3rd[sampled_pos_inds_subset],
        reduction='none',
    )
    loss_rcnn_reg_3rd = (loss_rcnn_reg_3rd / box_labels_3rd.numel()).sum()

    return dict(
        loss_rcnn_cls_1st=loss_rcnn_cls_1st,
        loss_rcnn_reg_1st=loss_rcnn_reg_1st,
        loss_rcnn_cls_2nd=loss_rcnn_cls_2nd,
        loss_rcnn_reg_2nd=loss_rcnn_reg_2nd,
        loss_rcnn_cls_3rd=loss_rcnn_cls_3rd,
        loss_rcnn_reg_3rd=loss_rcnn_reg_3rd,
    )
