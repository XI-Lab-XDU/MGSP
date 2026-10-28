# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import torch
from PIL import Image
import os.path as osp
import json
import os
# SAVE_PATH="/data/zhy/COAT/data/PRW/label2instance"

class BaseDataset:
    """
    Base class of person search dataset.
    """

    def __init__(self, root, transforms, split):
        self.root = root
        # self.text_root = osp.join(self.root, 'text_feature_resnet_instance') # modify annos here 
        # self.image_root = osp.join(self.root, 'image_feature_resnet_instance') 
        # self.global_text_root = osp.join(self.root, 'text_feature_resnet_global') 
        # self.global_image_root = osp.join(self.root, 'image_feature_resnet_global') 
        #
        self.text_root = osp.join(self.root, 'text_feature_resnet_instance_new') # modify annos here  SOTA用的这里
        self.image_root = osp.join(self.root, 'image_feature_resnet_instance_new') 
        self.global_text_root = osp.join(self.root, 'text_feature_resnet_global_new') 
        self.global_image_root = osp.join(self.root, 'image_feature_resnet_global_new') 
        #

        # self.text_root = osp.join(self.root, 'text_feature_resnet_instance_7b') # modify annos here 
        # self.image_root = osp.join(self.root, 'image_feature_resnet_instance_7b') 
        # self.global_text_root = osp.join(self.root, 'text_feature_resnet_global_7b') 
        # self.global_image_root = osp.join(self.root, 'image_feature_resnet_global_7b') 

        self.transforms = transforms
        self.split = split
        self.start = 0
        self.unlabel_start = 5555
        assert self.split in ("train", "gallery", "query")
        # self.label2instance = {}
        # self.unlabel_count=5555
        self.annotations = self._load_annotations()
        # with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        #     json.dump(self.label2instance, f, ensure_ascii=False, indent=4)


    def _load_annotations(self):
        """
        For each image, load its annotation that is a dictionary with the following keys:
            img_name (str): image name
            img_path (str): image path
            boxes (np.array[N, 4]): ground-truth boxes in (x1, y1, x2, y2) format
            pids (np.array[N]): person IDs corresponding to these boxes
            cam_id (int): camera ID (only for PRW dataset)
        """
        raise NotImplementedError

    # def __getitem__(self, index): 失败案例
    #     anno = self.annotations[index]
    #     img = Image.open(anno["img_path"]).convert("RGB")
    #     boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
    #     labels = torch.as_tensor(anno["pids"], dtype=torch.int64)
    #     org_shapes = img.size # 新加的
    #     image_feaetures = []
    #     text_features = []
    #     text_token = []
    #     text_attn = []
    #     text_pos = []
    #     if self.split == "train":
    #         for i in anno["instances_id"]:
    #             image_feaetures.append(torch.load(osp.join(self.image_root, f'{i}.pt'), weights_only=True))
    #             text_features.append(torch.load(osp.join(self.text_root, f'{i}.pt'), weights_only=True))
    #             text_token.append(torch.load(osp.join(self.token_root, f'{i}.pt'), weights_only=True))
    #             text_attn.append(torch.load(osp.join(self.attn_root, f'{i}.pt'), weights_only=True))
    #             text_pos.append(torch.load(osp.join(self.pos_root, f'{i}.pt'), weights_only=True))
    #         image_feaetures = torch.cat(image_feaetures)
    #         text_features = torch.cat(text_features)
    #         text_token = torch.cat(text_token)
    #         text_attn = torch.cat(text_attn)
    #         text_pos = torch.cat(text_pos)
    #     else:
    #         image_feaetures=torch.zeros(1)
    #         text_features=torch.zeros(1)
    #         text_token = torch.zeros(1)
    #         text_attn = torch.zeros(1)
    #         text_pos = torch.zeros(1)
    #     instances_id = torch.as_tensor(anno["instances_id"], dtype=torch.int64)
    #     # target = {"img_name": anno["img_name"], "boxes": boxes, "labels": labels, "instances_id": instances_id, "image_features":image_feaetures, "text_features":text_features}
    #     target = {"img_name": anno["img_name"], "boxes": boxes, "org_shapes": org_shapes, "labels": labels, "instances_id": instances_id, "image_features":image_feaetures, "text_features":text_features, "text_token":text_token, "text_attn":text_attn, "text_pos":text_pos}
    #     if self.transforms is not None:
    #         img, target = self.transforms(img, target)
    #     return img, target


    def __getitem__(self, index):
        anno = self.annotations[index]
        img = Image.open(anno["img_path"]).convert("RGB")
        boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(anno["pids"], dtype=torch.int64)
        image_feaetures = []
        text_features = []
        global_image_features = []
        global_text_features = []
        if self.split == "train":
            for i in anno["instances_id"]:
                image_feaetures.append(torch.load(osp.join(self.image_root, f'{i}.pt'), weights_only=True))
                text_features.append(torch.load(osp.join(self.text_root, f'{i}.pt'), weights_only=True))
            prefix, _ = os.path.splitext(anno['img_name'])
            global_image_features.append(torch.load(osp.join(self.global_image_root, f'{prefix}.pt'), weights_only=True))
            global_text_features.append(torch.load(osp.join(self.global_text_root, f'{prefix}.pt'), weights_only=True))

            image_feaetures = torch.cat(image_feaetures)
            text_features = torch.cat(text_features)
            global_image_features = torch.cat(global_image_features)
            global_text_features = torch.cat(global_text_features)
        else:
            image_feaetures=torch.zeros(1)
            text_features=torch.zeros(1)
            global_image_features = torch.zeros(1)
            global_text_features = torch.zeros(1)
        instances_id = torch.as_tensor(anno["instances_id"], dtype=torch.int64)
        target = {"img_name": anno["img_name"], "boxes": boxes, "labels": labels, "instances_id": instances_id, 
                  "image_features":image_feaetures, "text_features":text_features, "global_image_features":global_image_features, "global_text_features":global_text_features,}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.annotations)