# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

from collections import OrderedDict
import torch.nn.functional as F
import torchvision
from torch import nn
import torch
from .clip import clip
import math
from .backbone import backbone
from detectron2.layers import CNNBlockBase, Conv2d, get_norm
from detectron2.modeling.backbone.fpn import _assert_strides_are_log2_contiguous
import copy
import loralib as lora
from open_clip.transformer import ResidualAttentionBlock
from typing import Optional

class SimAM(torch.nn.Module):
    def __init__(self, channels = None, e_lambda = 1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):

        b, c, h, w = x.size()
        
        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2,3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.GELU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.GELU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )
            elif pool_type=='lp':
                lp_pool = F.lp_pool2d( x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( lp_pool )
            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale


class MLP(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = num_features * expansion_factor
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        x = self.dropout2(self.fc2(x))
        return x


class TokenMixer(nn.Module):
    def __init__(self, num_features, image_size ,num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_patches, expansion_factor, dropout)
        self.image_size = image_size        
        self.spatial_att = SpatialGate()

    def SpatialGate_forward(self, x):
        residual =x 
        BB, HH_WW, CC = x.shape
        HH =  WW = int(math.sqrt(HH_WW))
        x = x.reshape(BB, CC, HH, WW)
        x = self.spatial_att(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(BB, -1, CC)
        x = residual + x    
        return x

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x_pre_norm = self.SpatialGate_forward((x))
        x = self.norm(x_pre_norm)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_features, num_patches)
        x = self.mlp(x)
        x = x.transpose(1, 2)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class ChannelMixer(nn.Module):
    def __init__(self, num_features, image_size, num_patches, expansion_factor, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)
        self.mlp = MLP(num_features, expansion_factor, dropout)
        self.image_size =image_size
        self.channel_att = ChannelGate(num_features, )

    def ChannelGate_forward(self, x):
        residual =x 
        BB, HH_WW, CC = x.shape
        HH =  WW = int(math.sqrt(HH_WW))       
        x = x.reshape(BB, CC, HH, WW)
        x = self.channel_att(x)
        x = x.permute(0, 2, 3, 1)
        x = x.view(BB, -1, CC)
        x = residual + x    
        return x

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        residual = x
        x_pre_norm = self.ChannelGate_forward(x)
        x = self.norm(x_pre_norm)
        x = self.mlp(x)
        # x.shape == (batch_size, num_patches, num_features)
        out = x + residual
        return out


class MixerLayer(nn.Module):
    def __init__(self, num_features, image_size, num_patches, expansion_factor, dropout):
        super().__init__()
        self.token_mixer = TokenMixer(
            num_patches, image_size, num_features, expansion_factor, dropout
        )
        self.channel_mixer = ChannelMixer(
            num_patches, image_size, num_features, expansion_factor, dropout
        )

    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        x = self.token_mixer(x)
        x = self.channel_mixer(x)
        # x.shape == (batch_size, num_patches, num_features)
        return x


def check_sizes(image_size, patch_size):
    sqrt_num_patches, remainder = divmod(image_size, patch_size)
    assert remainder == 0, "`image_size` must be divisibe by `patch_size`"
    num_patches = sqrt_num_patches ** 2
    return num_patches


class ARM_Mixer(nn.Module):
    def __init__(
        self,
        image_size=14,
        patch_size=1,
        in_channels=256,
        num_features=256,
        expansion_factor=3,        
        dropout=0.3,
    ):
        num_patches = check_sizes(image_size, patch_size)
        super().__init__()
        
        self.mixers = MixerLayer(num_patches, image_size, num_features, expansion_factor, dropout)
        self.simam = SimAM()

    def forward(self, x):
        residual = x
        BB, CC, HH, WW= x.shape
        patches = x.permute(0, 2, 3, 1)
        patches = patches.view(BB, -1, CC)
        # patches.shape == (batch_size, num_patches, num_features)
        embedding = self.mixers(patches)
        
        embedding_rearrange = embedding.reshape(BB, CC,HH,WW)
        embedding_final = embedding_rearrange + self.simam(x)+x
        return embedding_final


class Backbone(nn.Sequential):
    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


class Res5Head(nn.Sequential):
    def __init__(self, resnet):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["before_trans", x], ["after_trans", feat]])



def load_clip_to_cpu(backbone_name, h_resolution, w_resolution):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution)

    return model

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.tokenized_prompts = clip_model.token_embedding
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, tokenized_prompts):
        prompts = self.tokenized_prompts(tokenized_prompts)
        x = prompts + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class PromptLearner(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        ctx_init = "a photo of a"
        ctx_dim = 512
        dtype = clip_model.dtype
        prompt_prefix = ctx_init
        
        n_ctx = 4
        # prompt = clip.tokenize(ctx_init)
        # with torch.no_grad():
        #     embedding = clip_model.token_embedding(prompt).type(dtype)
        # ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        ctx_vectors = torch.empty(482, 4, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)
        
        
        prompt = prompt_prefix + " " + "person" + "."
        tokenized_prompts = clip.tokenize(prompt)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype) 
        
        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  
        # self.num_class = num_class
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        
        # self.meta_net = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(512, 512)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(512, ctx_dim))
        # ]))
        
    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,     # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self, label):
        
        ctx = self.ctx[label]
        prefix = self.token_prefix
        suffix = self.token_suffix
        # ctx = self.ctx
        # bias = self.meta_net(im_features)
        # bias = bias.unsqueeze(1)
        # ctx = ctx.unsqueeze(0)
        
        b = ctx.shape[0]
        prefix = self.token_prefix.expand(b, -1, -1) 
        suffix = self.token_suffix.expand(b, -1, -1) 
            
        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                suffix,  # (n_cls, *, dim)
            ],
            dim=1,
        ) 

        return prompts 

 
class ClipResNet(nn.Module):
    def __init__(self, clip_model):
        super(ClipResNet, self).__init__()
        self.stem = nn.Sequential(
            clip_model.conv1,
            clip_model.bn1,
            clip_model.act1,
            clip_model.conv2,
            clip_model.bn2,
            clip_model.act2,
            clip_model.conv3,
            clip_model.bn3,
            clip_model.act3,
            clip_model.avgpool,
        )
        self.layer1 = clip_model.layer1
        self.layer2 = clip_model.layer2
        self.layer3 = clip_model.layer3
        self.layer4 = clip_model.layer4
        self.out_channels = 1024
        
        for param in self.stem.parameters():
            param.requires_grad = False


    def forward(self, x):
        # using the forward method from nn.Sequential
        
        x = self.stem(x)
        feat_res2 = self.layer1(x)  # 256 channels
        feat_res3 = self.layer2(feat_res2)  # 512 channels
        feat_res4 = self.layer3(feat_res3)  # 1024 channels
        feat_res5 = self.layer4(feat_res4)  # 2048 channels
        return [feat_res3, feat_res4, feat_res5]
  
class ClipBackbone(nn.Module):
    def __init__(self, clip_model):
        super(ClipBackbone, self).__init__()
        # super(Backbone, self).__init__(
        #     OrderedDict(
        #         [
        #             ["conv1", clip_model.conv1],
        #             ["bn1", clip_model.bn1],
        #             ["relu", clip_model.relu],
        #             # ["maxpool", resnet.maxpool],
        #             ["layer1", clip_model.layer1],  # res2
        #             ["layer2", clip_model.layer2],  # res3
        #             ["layer3", clip_model.layer3],  # res4
        #         ]
        #     )
        # )
        self.stem = nn.Sequential(
            clip_model.conv1,
            clip_model.bn1,
            clip_model.act1,
            clip_model.conv2,
            clip_model.bn2,
            clip_model.act2,
            clip_model.conv3,
            clip_model.bn3,
            clip_model.act3,
            clip_model.avgpool,
            # clip_model.layer1
        )
        self.layer1 = clip_model.layer1
        self.layer2 = clip_model.layer2
        self.layer3 = clip_model.layer3
        # self.layer4 = clip_model.layer4
        self.out_channels = 1024
        
        for param in self.stem.parameters():
            param.requires_grad = False


    def forward(self, x):
        # using the forward method from nn.Sequential
        # x = x.type(self.stem[0].weight.dtype) 
        x = self.stem(x)
        # feat = super(Backbone, self).forward(x)
        x = self.layer1(x) # ->256
        x = self.layer2(x) # ->512
        x = self.layer3(x) # ->1024
        # x = self.layer4(x)
        return OrderedDict([["feat_res4", x]])


class ClipRes5Head(nn.Module):
    def __init__(self, clip_model):
        super(ClipRes5Head, self).__init__()
        # super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.layer4 = clip_model.layer4
        # self.attnpool = clip_model.attnpool
        self.out_channels = [1024, 1024]
        # for param in self.parameters():
        #     param.requires_grad = False
        # self.mlP_model = ARM_Mixer(in_channels=256, image_size=14, patch_size=1)
        # self.qconv1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
        # self.qconv2 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=1)
        # for param in self.attnpool.parameters():
        #     param.requires_grad = False
            
    def forward(self, x):
        # x = self.qconv1(x)
        # x=self.mlP_model(x)
        # x = self.qconv2(x)
        # feat = super(Res5Head, self).forward(x)
        # x = x.type(self.layer4[0].conv1.weight.dtype)
        feat = self.layer4(x) # [512,1024,14,14] -> [512, 2048, 7, 7]
        
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1).flatten(1)
        # feat = self.attnpool(feat)
        
        return OrderedDict([["before_trans", x], ["after_trans", feat]])

class ClipPromptRes5Head(nn.Module):
    def __init__(self, clip_model):
        super(ClipPromptRes5Head, self).__init__()
        # super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.layer4 = clip_model.visual.layer4
        # self.attnpool = clip_model.visual.attnpool
        # self.text_encoder = TextEncoder(clip_model)
        self.out_channels = [1024, 2048]
        # self.prompt_learner = PromptLearner(clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # self.logit_scale = clip_model.logit_scale

    def forward(self, x):
        # feat = super(Res5Head, self).forward(x)
        # tokenized_prompts = self.tokenized_prompts
        # logit_scale = self.logit_scale.exp()
        # x = x.type(self.layer4[0].conv1.weight.dtype)
        # feat = self.layer4(x)
        feat = self.attnpool(x)
        x = F.adaptive_max_pool2d(x, 1)
        
        # prompts = self.prompt_learner(feat)
        
        # text_feature = self.text_encoder(prompts, self.tokenized_prompts)
        return OrderedDict([["before_trans", x], ["after_trans", feat]])


# Convnext Backbone
class ConvnextBackbone(nn.Module):
    def __init__(self, convnext):
        super().__init__()
        self.stem = nn.Sequential(
            convnext.trunk.stem,  
        )
        self.body = nn.Sequential(
            convnext.trunk.stages[1],
            convnext.trunk.stages[2],
          )
        self.out_channels = 512
        for param in self.stem.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        x = self.stem(x)
        x = self.body(x)
        return OrderedDict([["feat_res4", x]])
      


# Convnext Head
class ConvnextHead(nn.Module):
    def __init__(self, convnext):
        super().__init__()
        self.head = convnext.trunk.stages[3]
        # self.pool_conv = nn.Sequential(
        #     convnext.trunk.norm_pre,
        #     convnext.trunk.head,
        #     convnext.head
        # )
        self.out_channels = [
            512,
            1024,
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']
    
    def forward(self, x):
        feat = self.head(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        
        # prompts = self.prompt_learner(feat)
        
        # text_feature = self.text_encoder(prompts, self.tokenized_prompts)
        return OrderedDict([["before_trans", x], ["after_trans", feat]])
    
def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)

class VitBackbone(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.conv1 = vit.conv1
        self.class_embedding = vit.class_embedding
        self.positional_embedding = vit.positional_embedding
        self.patch_dropout = vit.patch_dropout
        self.ln_pre = vit.ln_pre
        
        self.body = nn.Sequential(
            *vit.transformer.resblocks[:8],
          )
        # self.body = nn.Sequential(
        #     *vit.transformer.lora_resblocks[:8], # 每个block增加了2个lora参数
        #   )
        # self.body = nn.Sequential(
        #     *vit.transformer.adapter_resblocks[:10], # 每个block增加了4个adapter参数
        #   )
        self.out_channels = 768
        self.class_embedding.requires_grad = False  # class_embedding 可能是 nn.Parameter
        self.positional_embedding.requires_grad = False  # positional_embedding 可能是 nn.Parameter
        
        params = {name: param.detach().clone() for name, param in self.named_parameters()}
        # for param in self.body.parameters(): # 会减少显存消耗
        #     param.requires_grad = False
        # for key, value in self.body.named_parameters(recurse=True):
        #     if 'lora_A' in key or 'lora_B' in key:
        #         value.requires_grad = True
        # for key, value in self.body.named_parameters(recurse=True):
        #     if 'adaptmlp' in key:
        #         value.requires_grad = True
        for param in self.patch_dropout.parameters():
            param.requires_grad = False
        for param in self.ln_pre.parameters():
            param.requires_grad = False
        for param in self.conv1.parameters():
            param.requires_grad = False
        print('Backbone created')
    
    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid] [6, 3, 1504, 1216] -> [6, 768, 94, 76]
        H_patch = x.shape[2]
        W_patch = x.shape[3]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # [6, 94*76, 768]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)
        
        # adaptive positional_emb
        pos_embed = self.positional_embedding
        cls_pos = pos_embed[0:1, :]
        patch_pos = pos_embed[1:, :] 
        num_patches = patch_pos.shape[0]
        orig_size = int(num_patches ** 0.5)
        assert orig_size * orig_size == num_patches, "pos_embed 不成方形" # 确保能开方
        patch_pos = patch_pos.reshape(orig_size, orig_size, -1).permute(2, 0, 1).unsqueeze(0) # reshape回14*14
        patch_pos = F.interpolate(patch_pos, size=(H_patch, W_patch), mode='bilinear', align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, H_patch * W_patch, -1).squeeze(0) 
        adaptive_pos_embed = torch.cat([cls_pos, patch_pos], dim=0)
        x = x + adaptive_pos_embed.to(x.dtype)
        
        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        
        x = self.body(x) # [8,7145,768] # break
        x = x[:, 1:]  # 去掉 cls token
        x = x.view(x.shape[0], H_patch, W_patch, x.shape[2]).permute(0, 3, 1, 2)
        return OrderedDict([["feat_res4", x]])

class VitHead(nn.Module):
    def __init__(self, vit):
        super().__init__()
        self.head = nn.Sequential(
            *vit.transformer.resblocks[8:],
          )
        # self.head = nn.Sequential(
        #     *vit.transformer.lora_resblocks[8:],
        #   )
        # self.head = nn.Sequential(
        #     *vit.transformer.adapter_resblocks[10:],
        #   )
        self.manu_conv = nn.Conv2d(
            in_channels=768,
            out_channels=1024, # 从1024改为512 因为用了正确的clip-vit
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(1024, 2048, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True)
        )
        self.out_channels = [
            512,
            1024,
        ]
        self.featmap_names = ['feat_res4', 'feat_res5']
        # for param in self.head.parameters(): # 会减少显存消耗
        #     param.requires_grad = False
        # for key, value in self.head.named_parameters(recurse=True):
        #     if 'lora_A' in key or 'lora_B' in key:
        #         value.requires_grad = True
        # for key, value in self.head.named_parameters(recurse=True):
        #     if 'adaptmlp' in key:
        #         value.requires_grad = True
        print('Head Created')
        
    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # [128*b, 14*14, 768]
        feat = self.head(x) # break
        x = x.view(x.shape[0], 14, 14, x.shape[2]).permute(0, 3, 1, 2)
        feat = feat.view(feat.shape[0], 14, 14, feat.shape[2]).permute(0, 3, 1, 2)
        x = self.manu_conv(x)
        feat = self.manu_conv(feat) # 手动改维度
        feat = self.pool(feat)
        feat = self.conv(feat)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1).flatten(1)
        
        # prompts = self.prompt_learner(feat)
        
        # text_feature = self.text_encoder(prompts, self.tokenized_prompts)
        return OrderedDict([["before_trans", x], ["after_trans", feat]]) # 这里的before_trans有什么用 break

def build_trans_resnet(clip_model):
    
    return ClipResNet(clip_model.visual)
def build_clip_resnet(clip_model):
    # resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # # freeze layers
    # resnet.conv1.weight.requires_grad_(False)
    # resnet.bn1.weight.requires_grad_(False)
    # resnet.bn1.bias.requires_grad_(False)

    return ClipBackbone(clip_model.visual), ClipRes5Head(clip_model.visual)

def build_resnet(name="resnet50", pretrained=True):
    resnet = torchvision.models.resnet.__dict__[name](pretrained=pretrained)

    # freeze layers
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)
  
def build_convext(clip_model):

    return ConvnextBackbone(clip_model.visual), ConvnextHead(clip_model.visual)

def build_vit(clip_model):
    # build_lora_resblocks(clip_model.visual) # 加入lora
    # build_adapter_resblocks(clip_model.visual)
    return VitBackbone(clip_model.visual), VitHead(clip_model.visual)


def build_lora_resblocks(vit_transformer, r=4, lora_alpha=16):
    lora_resblocks = []

    for block in vit_transformer.transformer.resblocks:
        # 深拷贝原 block，避免修改原始权重
        new_block = copy.deepcopy(block)

        # 插入 LoRA 到 attn.out_proj
        old_proj = new_block.attn.out_proj
        lora_proj = lora.Linear(
            old_proj.in_features,
            old_proj.out_features,
            r=r,
            lora_alpha=lora_alpha,
            bias=True
        )
        lora_proj.weight.data = old_proj.weight.data.clone()
        lora_proj.bias.data = old_proj.bias.data.clone()
        new_block.attn.out_proj = lora_proj

        # 可选：插入 LoRA 到 MLP 层
        # mlp_fc = new_block.mlp[0]
        # mlp_proj = new_block.mlp[2]
        # ...

        lora_resblocks.append(new_block)

    # 保存到 transformer.lora_resblocks 中
    vit_transformer.transformer.lora_resblocks = nn.ModuleList(lora_resblocks)


# 构建 adapter 版本的 resblocks
def build_adapter_resblocks(vit_model):
    adapter_blocks = []
    d_model = 768
    n_head = 12 # 这里为什么也是12?
    layers = 12 # 这里是12个block
    # mlp_ratio = 4.0,
    # ls_init_value = None,
    # is_cross_attention = False,
    # batch_first = True,
    # act_layer = nn.GELU,
    # norm_layer = LayerNorm,
    
    # for block in vit_model.transformer.resblocks:
    #     adapter_block = AdapterResBlock(block)
    #     adapter_blocks.append(adapter_block)
    
    adapter_blocks = nn.ModuleList([
            AdapterResBlock(d_model, n_head)
            for _ in range(layers)
        ])
    vit_model.transformer.adapter_resblocks = nn.ModuleList(adapter_blocks)


class AdapterResBlock(ResidualAttentionBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 构建 adapter
        adapter_cfg = dict( # 手动加的配置
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=256, # 原来是64现在改为256
            d_model=768, # 输入的维度
        )
        adapter_cfg = AdapterConfig(**adapter_cfg)
        self.adapter_cfg = adapter_cfg
        if adapter_cfg.ffn_adapt:
            self.adaptmlp = Adapter(self.adapter_cfg, 
                                    d_model=adapter_cfg.d_model, 
                                    bottleneck=adapter_cfg.ffn_num, 
                                    dropout=0.1, 
                                    init_option=adapter_cfg.ffn_adapter_init_option,
                                    adapter_scalar=adapter_cfg.ffn_adapter_scalar,
                                    adapter_layernorm_option=adapter_cfg.ffn_adapter_layernorm_option,
                                    )
    
    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        # ============ Attention Block ============ #
        residual = q_x
        q = self.ln_1(q_x)
        attn_out = self.attention(q_x=q, k_x=k_x, v_x=v_x, attn_mask=attn_mask)
        x = residual + self.ls_1(attn_out) # 原本有个layer_scale不知道是什么
        # ============ MLP Block ============ #
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + self.ls_2(x)
        # ============ Adapter Block ============ #
        x = x + self.adaptmlp(residual, add_residual=False)
        return x
    
class AdapterResBlock1(nn.Module):
    def __init__(self, original_block: nn.Module):
        super().__init__()
        # 复用原始 block 的组件（注意：这里是引用，不是 deepcopy，保证原参数不变）
        self.ln_1 = original_block.ln_1
        self.attn = original_block.attn
        self.ln_2 = original_block.ln_2
        self.mlp = original_block.mlp
        # 构建 adapter
        adapter_cfg = dict( # 手动加的配置
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=256, # 原来是64现在改为256
            d_model=768, # 输入的维度
        )
        adapter_cfg = AdapterConfig(**adapter_cfg)
        self.adapter_cfg = adapter_cfg
        if adapter_cfg.ffn_adapt:
            self.adaptmlp = Adapter(self.adapter_cfg, 
                                    d_model=adapter_cfg.d_model, 
                                    bottleneck=adapter_cfg.ffn_num, 
                                    dropout=0.1, 
                                    init_option=adapter_cfg.ffn_adapter_init_option,
                                    adapter_scalar=adapter_cfg.ffn_adapter_scalar,
                                    adapter_layernorm_option=adapter_cfg.ffn_adapter_layernorm_option,
                                    )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))[0]
        identity = x
        x = self.mlp(self.ln_2(x)) # 这里是不是少了一个residual
        x = x + self.adapter(identity, add_residual=False)
        return x
    
class Adapter(nn.Module):
    def __init__(self,
                 config=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        #_before
        self.adapter_layernorm_option = adapter_layernorm_option

        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)

        self.dropout = dropout
        if init_option == "bert":
            raise NotImplementedError
        elif init_option == "lora":
            with torch.no_grad():
                nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
                nn.init.zeros_(self.up_proj.weight)
                nn.init.zeros_(self.down_proj.bias)
                nn.init.zeros_(self.up_proj.bias)

    def forward(self, x, add_residual=True, residual=None):
        residual = x if residual is None else residual
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)

        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = nn.functional.dropout(down, p=self.dropout, training=self.training) # 用了0.1的drop out这里可以调
        up = self.up_proj(down)

        up = up * self.scale # scale也是可以调的参数

        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm_before(up)

        if add_residual:
            output = up + residual
        else:
            output = up

        return output    

class AdapterConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            
class SimpleFeaturePyramid(backbone): # 这里的backbone是vitdet里来的，和其他的不一样
    """
    This module modifies SimpleFeaturePyramid in :paper:`vitdet`.
    It creates pyramid features built on top of the input feature map.
    """

    def __init__(
        self,
        input_shapes,
        in_feature,
        out_channels,
        scale_factors,
        top_block=None,
        norm="LN",
        square_pad=0,
    ):
        """
        Args:
            net (Backbone): module representing the subnetwork backbone.
                Must be a subclass of :class:`Backbone`.
            in_feature (str): names of the input feature maps coming
                from the net.
            out_channels (int): number of channels in the output feature maps.
            scale_factors (list[float]): list of scaling factors to upsample or downsample
                the input features for creating pyramid features.
            top_block (nn.Module or None): if provided, an extra operation will
                be performed on the output of the last (smallest resolution)
                pyramid output, and the result will extend the result list. The top_block
                further downsamples the feature map. It must have an attribute
                "num_levels", meaning the number of extra pyramid levels added by
                this block, and "in_feature", which is a string representing
                its input feature (e.g., p5).
            norm (str): the normalization to use.
            square_pad (int): If > 0, require input images to be padded to specific square size.
        """
        super(SimpleFeaturePyramid, self).__init__()

        self.scale_factors = scale_factors

        strides = [
            int(input_shapes[in_feature].stride / scale) for scale in scale_factors
        ]
        _assert_strides_are_log2_contiguous(strides)

        dim = input_shapes[in_feature].channels
        self.stages = []
        use_bias = norm == ""
        for idx, scale in enumerate(scale_factors):
            out_dim = dim
            if scale == 4.0:
                layers = [
                    nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2),
                    get_norm(norm, dim // 2),
                    nn.GELU(),
                    nn.ConvTranspose2d(dim // 2, dim // 4, kernel_size=2, stride=2),
                ]
                out_dim = dim // 4
            elif scale == 2.0:
                layers = [nn.ConvTranspose2d(dim, dim // 2, kernel_size=2, stride=2)]
                out_dim = dim // 2
            elif scale == 1.0:
                layers = []
            elif scale == 0.5:
                layers = [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                raise NotImplementedError(f"scale_factor={scale} is not supported yet.")

            layers.extend(
                [
                    Conv2d(
                        out_dim,
                        out_channels,
                        kernel_size=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                    Conv2d(
                        out_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=use_bias,
                        norm=get_norm(norm, out_channels),
                    ),
                ]
            )
            layers = nn.Sequential(*layers)

            stage = int(math.log2(strides[idx]))
            self.add_module(f"simfp_{stage}", layers)
            self.stages.append(layers)

        self.in_feature = in_feature
        self.top_block = top_block
        # Return feature names are "p<stage>", like ["p2", "p3", ..., "p6"]
        self._out_feature_strides = {
            "p{}".format(int(math.log2(s))): s for s in strides
        }
        # top block output feature maps.
        if self.top_block is not None:
            for s in range(stage, stage + self.top_block.num_levels):
                self._out_feature_strides["p{}".format(s + 1)] = 2 ** (s + 1)

        self._out_features = list(self._out_feature_strides.keys())
        self._out_feature_channels = {k: out_channels for k in self._out_features}
        self._size_divisibility = strides[-1]
        self._square_pad = square_pad

    @property
    def padding_constraints(self):
        return {
            "size_divisiblity": self._size_divisibility,
            "square_size": self._square_pad,
        }

    def forward(self, x):
        """
        Args:
            x: input feature map.

        Returns:
            dict[str->Tensor]:
                mapping from feature map name to pyramid feature map tensor
                in high to low resolution order. Returned feature names follow the FPN
                convention: "p<stage>", where stage has stride = 2 ** stage e.g.,
                ["p2", "p3", ..., "p6"].
        """
        bottom_up_features = x
        features = bottom_up_features[self.in_feature] # 用的是stage3而不是side_stage3,为什么这里用的是reid的特征
        results = []

        for stage in self.stages: # 经历了四个stage
            results.append(stage(features))

        if self.top_block is not None:
            if self.top_block.in_feature in bottom_up_features:
                top_block_in_feature = bottom_up_features[self.top_block.in_feature]
            else: # 执行了这里
                top_block_in_feature = results[
                    self._out_features.index(self.top_block.in_feature)
                ]
            results.extend(self.top_block(top_block_in_feature))
        assert len(self._out_features) == len(results)
        return {f: res for f, res in zip(self._out_features, results)}