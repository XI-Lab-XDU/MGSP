# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import torch
import torch.nn.functional as F
from torch import autograd, nn

class OIM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq)
        ctx.header = header
        ctx.momentum = momentum
        ctx.dtype = inputs.dtype
        
        inputs = inputs.to(torch.float32)
        lut = lut.to(torch.float32)
        cq = cq.to(torch.float32)

        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())

        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1).to(ctx.dtype)

    @staticmethod
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq = ctx.saved_tensors
        header = ctx.header
        momentum = ctx.momentum
        dtype = ctx.dtype

        grad_outputs = grad_outputs.to(torch.float32)
        inputs = inputs.to(torch.float32)
        lut = lut.to(torch.float32)
        cq = cq.to(torch.float32)

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))
            grad_inputs = grad_inputs.to(dtype)

        for x, y in zip(inputs, targets):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    header_tensor = torch.tensor(header, dtype=torch.long, device=inputs.device)
    momentum_tensor = torch.tensor(momentum, dtype=torch.float32, device=inputs.device)
    return OIM.apply(inputs, targets, lut, cq, header_tensor, momentum_tensor)



class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, un_oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.un_oim_scalar = un_oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0

    @torch.no_grad()
    def memory_update(self, inputs, targets):
        for x, y in zip(inputs, targets):
            if y < len(self.lut):
                self.lut[y] = self.momentum * self.lut[y] + (1.0 - self.momentum) * x
                self.lut[y] /= self.lut[y].norm()
            else:
                self.cq[self.header_cq] = x
                self.header_cq = (self.header_cq + 1) % self.cq.size(0)


    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = label >= 0
        
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        # projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        outputs_labeled = inputs.mm(self.lut.t().clone()) * self.oim_scalar
        outputs_unlabeled = inputs.mm(self.cq.t().clone()) * self.un_oim_scalar
        # projected - Tensor [M, lut+cq], e.g., [M, 482+500]=[M, 982]
        projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        
        # projected *= self.oim_scalar        

        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        
        if (label == 5554).all():
            loss_oim = inputs.sum() * 0
        else:
            loss_oim = F.cross_entropy(projected, label, ignore_index=5554)
        
        self.memory_update(inputs, label)
        
        return loss_oim, inputs, label

class OIMLossSafe(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar, use_cq=True):
        super(OIMLossSafe, self).__init__()
        # Store params
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar
        self.ignore_index = num_pids
        self.margin = 0.15 # cuhk:0.25, prw:0.15, g2aps:0.15

        # Setup buffers
        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        if use_cq:
            self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))
            self.header_cq = 0
        else:
            self.header_cq = 0
            self.cq = None
        self.triplet = TripletCosineLoss(self.margin)
            
    def _triplet_loss(self, l_vectors, l_labels, u_vectors):
        m_labels = l_labels.unique()
        m_vectors = self.lut[m_labels]
        return self.triplet(
            anchor_vectors=torch.cat([l_vectors, m_vectors]),
            anchor_labels=torch.cat([l_labels, m_labels]),
            sample_vectors=torch.cat([l_vectors, u_vectors.detach(), m_vectors]),
            sample_labels=torch.cat([l_labels, l_labels.new_full((len(u_vectors),), -100), m_labels]),
            reduction='mean',
        )
        
    def forward(self, inputs, roi_label, instance_ids):
        # Normalize inputs
        roi_label = torch.cat(roi_label)
        label = roi_label - 1  # background label = -1

        inds = label >= 0
        
        label = label[inds]
        inputs = inputs[inds]
        instance_ids = torch.cat(instance_ids)[inds]
        label[label >= 5554] = 5554
        
        # inputs_origin = inputs
        
        inputs = F.normalize(inputs.view(-1, self.num_features), dim=1)

        # Compute masks to avoid using unfilled entries in LUT, CQ
        with torch.no_grad():
            bad_lut_mask = torch.all(self.lut == 0, dim=1)
            bad_lut_idx = torch.where(bad_lut_mask)[0]
            bad_pos_mask = (label.unsqueeze(1) == bad_lut_idx).any(dim=1)
            bad_label = label[bad_pos_mask]
            bad_pos_idx = torch.where(bad_pos_mask)[0]
            bad_cq_mask = torch.all(self.cq == 0, dim=1)

        # Compute cosine similarity of inputs with LUT
        outputs_labeled = inputs.mm(self.lut.t().clone())
        outputs_labeled[:, bad_lut_mask] = -1
        outputs_labeled[bad_pos_idx, bad_label] = 1

        # Compute cosine similarity of inputs with CQ
        if self.cq is not None:
            outputs_unlabeled = inputs.mm(self.cq.t().clone())
            outputs_unlabeled[:, bad_cq_mask] = -1
            projected = torch.cat([outputs_labeled, outputs_unlabeled], dim=1)
        else:
            projected = outputs_labeled

        # Multiply projections by (inverse) temperature scalar
        
        for i in range(len(label)):
            y = label[i]
            if y != 5554:
                projected[i, y] -= self.margin
        
        projected *= self.oim_scalar

        # Compute loss
        ## for numerical stability with float16, we divide before computing the sum to compute the mean
        ## WARNING: this may lead to underflow, experimental results give different result for this vs. mean reduce
        _loss_oim = F.cross_entropy(projected, label, ignore_index=5554, reduction='none')
        loss_oim = (_loss_oim / _loss_oim.size(0)).sum()

        # Compute LUT and CQ updates
        # unlabel_dict = {}
        with torch.no_grad():
            targets = label
            for x, y in zip(inputs, targets):
                if y < len(self.lut):
                    self.lut[y] = F.normalize(self.momentum * self.lut[y] + (1.0 - self.momentum) * x, dim=0)
                elif self.cq is not None:
                    self.cq[self.header_cq] = x
                    self.header_cq = (self.header_cq + 1) % self.cq.size(0)
        
        mask = label<len(self.lut)
        l_vectors, l_labels, u_vectors = inputs[mask], label[mask], inputs[~mask]
        loss_triplet = self._triplet_loss(l_vectors, l_labels, u_vectors)

        return loss_oim, loss_triplet

class TripletCosineLoss(nn.Module):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(
            self,
            anchor_vectors,
            anchor_labels,
            sample_vectors,
            sample_labels,
            reduction: str = "mean",
    ):
        cosine_sim_mat = anchor_vectors @ sample_vectors.t()
        labels_sam_mat = anchor_labels[:, None] == sample_labels
        positive, negative = [], []
        for cosine_sim, labels_sam in zip(cosine_sim_mat, labels_sam_mat):
            positive.append(torch.amin(cosine_sim[labels_sam]))  # Min cosine similarity in the same labels
            negative.append(torch.amax(cosine_sim[~labels_sam]))  # Max cosine similarity in the different labels
        return F.margin_ranking_loss(
            input1=torch.stack(positive),
            input2=torch.stack(negative),
            target=torch.ones_like(anchor_labels),
            margin=self.margin,
            reduction=reduction,
        )