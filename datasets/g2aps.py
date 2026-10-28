# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import os.path as osp
import numpy as np
from scipy.io import loadmat
from .base import BaseDataset
from PIL import Image
class G2APS(BaseDataset):
    def __init__(self, root, transforms, split):
        self.name = "G2APS"
        self.img_prefix = osp.join(root, "Image", "SSM")
        super(G2APS, self).__init__(root, transforms, split)

    def _load_queries(self):
        # TestG50: a test protocol, 50 gallery images per query
        protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
        protoc = protoc["TestG50"].squeeze()
        queries = []
        for item in protoc["Query"]:
            img_name = str(item["imname"][0, 0][0])
            roi = item["idlocate"][0, 0][0].astype(np.int32)
            roi[2:] += roi[:2]
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([-100]),  # dummy pid
                    "instances_id": [0]
                }
            )
        return queries

    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        # gallery images
        gallery_imgs = loadmat(osp.join(self.root, "annotation", "pool.mat"))
        gallery_imgs = gallery_imgs["pool"].squeeze()
        gallery_imgs = [str(a[0]) for a in gallery_imgs]
        if self.split == "gallery":
            return gallery_imgs
        # all images
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        all_imgs = [str(a[0][0]) for a in all_imgs]
        # training images = all images - gallery images
        training_imgs = sorted(list(set(all_imgs) - set(gallery_imgs)))
        return training_imgs

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        # load all images and build a dict from image to boxes
        all_imgs = loadmat(osp.join(self.root, "annotation", "Images.mat"))
        all_imgs = all_imgs["Img"].squeeze()
        name_to_boxes = {}
        name_to_pids = {}
        # self.unlabel_start = 5555  # default pid for unlabeled people
        for img_name, _, boxes in all_imgs:
            img_name = str(img_name[0])
            boxes = np.asarray([b[0] for b in boxes[0]])
            boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
            valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
            assert valid_index.size > 0, "Warning: {} has no valid boxes.".format(img_name)
            boxes = boxes[valid_index]
            name_to_boxes[img_name] = boxes.astype(np.int32)
            num_boxes = boxes.shape[0]
            # name_to_pids[img_name] = unlabeled_pid * np.ones(boxes.shape[0], dtype=np.int32)
            name_to_pids[img_name] = np.arange(self.unlabel_start, self.unlabel_start + num_boxes, dtype=np.int32)
            self.unlabel_start += num_boxes

        def set_box_pid(boxes, box, pids, pid):
            for i in range(boxes.shape[0]):
                if np.all(boxes[i] == box):
                    pids[i] = pid
                    return

        # assign a unique pid from 1 to N for each identity
        if self.split == "train":
            train = loadmat(osp.join(self.root, "annotation/test/train_test/Train.mat"))
            train = train["Train"].squeeze()
            for index, item in enumerate(train):
                scenes = item[0, 0][2].squeeze()
                for img_name, box, _ in scenes:
                    img_name = str(img_name[0])
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[img_name], box, name_to_pids[img_name], index + 1)
        else:
            protoc = loadmat(osp.join(self.root, "annotation/test/train_test/TestG50.mat"))
            protoc = protoc["TestG50"].squeeze()
            for index, item in enumerate(protoc):
                # query
                im_name = str(item["Query"][0, 0][0][0])
                box = item["Query"][0, 0][1].squeeze().astype(np.int32)
                set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)
                # gallery
                gallery = item["Gallery"].squeeze()
                for im_name, box, _ in gallery:
                    im_name = str(im_name[0])
                    if box.size == 0:
                        break
                    box = box.squeeze().astype(np.int32)
                    set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)

        annotations = []
        imgs = self._load_split_img_names()
        # cropped_dir = "/data2/zhx/LLM/data/G2APS/crop_image"  
        for img_name in imgs:
            boxes = name_to_boxes[img_name]
            boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)
            pids = name_to_pids[img_name]
            # cropped_paths = []
            # img_path = osp.join(self.img_prefix, img_name)
            # image = Image.open(img_path).convert("RGB")
            
            result = [self.start + i for i in range(len(boxes))]
            self.start = result[-1] + 1
                      
            # for i, (roi, res_id) in enumerate(zip(boxes, result)): # 这里在crop的时候没有用到，不知道会不会出问题
            #     # if label == 5555:
            #     #     self.unlabel_count+=1
            #     #     self.label2instance[self.unlabel_count] = [res_id]
            #     # elif label in self.label2instance:
            #     #     self.label2instance[int(label)].append(res_id)
            #     # else:
            #     #     self.label2instance[int(label)] = [res_id]
            #     x1, y1, x2, y2 = roi
            #     # Ensure coordinates are integers
            #     x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            #     # Optional: Validate ROI coordinates within image dimensions
            #     width, height = image.size
            #     x1 = max(0, min(x1, width - 1))
            #     y1 = max(0, min(y1, height - 1))
            #     x2 = max(x1 + 1, min(x2, width))
            #     y2 = max(y1 + 1, min(y2, height))

            #     # Crop the image using the ROI
            #     cropped_img = image.crop((x1, y1, x2, y2))

            #     # Define the cropped image name based on the unique instance ID
            #     cropped_name = f"{res_id}.jpg"  # You can retain the original extension if needed
            #     cropped_path = osp.join(cropped_dir, cropped_name)

            #     # Save the cropped image only if it doesn't already exist to avoid redundancy
            #     if not osp.exists(cropped_path):
            #         try:
            #             cropped_img.save(cropped_path)
            #         except IOError as e:
            #             print(f"Error saving cropped image {cropped_path}: {e}")
            #             continue  # Skip saving this cropped image if there's an error

            #   # # Append the path of the cropped image to the list
            #   # cropped_paths.append(cropped_path)            
            
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": boxes,
                    "pids": pids,
                    "instances_id": result
                }
            )
        return annotations
