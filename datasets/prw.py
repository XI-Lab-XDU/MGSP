# This file is part of COAT, and is distributed under the
# OSI-approved BSD 3-Clause License. See top-level LICENSE file or
# https://github.com/Kitware/COAT/blob/master/LICENSE for details.

import os.path as osp
import re

import numpy as np
from scipy.io import loadmat

from .base import BaseDataset
from PIL import Image

class PRW(BaseDataset):
    def __init__(self, root, transforms, split):
        self.name = "PRW"
        self.img_prefix = osp.join(root, "frames")
        super(PRW, self).__init__(root, transforms, split)

    def _get_cam_id(self, img_name):
        match = re.search(r"c\d", img_name).group().replace("c", "")
        return int(match)

    def _load_queries(self):
        query_info = osp.join(self.root, "query_info.txt")
        with open(query_info, "rb") as f:
            raw = f.readlines()

        queries = []
        for line in raw:
            linelist = str(line, "utf-8").split(" ")
            pid = int(linelist[0])
            x, y, w, h = (
                float(linelist[1]),
                float(linelist[2]),
                float(linelist[3]),
                float(linelist[4]),
            )
            roi = np.array([x, y, x + w, y + h]).astype(np.int32)
            roi = np.clip(roi, 0, None)  # several coordinates are negative
            img_name = linelist[5][:-2] + ".jpg"
            queries.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": roi[np.newaxis, :],
                    "pids": np.array([pid]),
                    "cam_id": self._get_cam_id(img_name),
                    "instances_id": [0]
                }
            )
        return queries

    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        assert self.split in ("train", "gallery")
        if self.split == "train":
            imgs = loadmat(osp.join(self.root, "frame_train.mat"))["img_index_train"]
        else:
            imgs = loadmat(osp.join(self.root, "frame_test.mat"))["img_index_test"]
        return [img[0][0] + ".jpg" for img in imgs]

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()
        # cropped_dir = "/data/zhy/COAT/data/PRW/crop_image"
        annotations = []
        imgs = self._load_split_img_names()
        for img_name in imgs:
            # img_path = osp.join(self.img_prefix, img_name)
            # image = Image.open(img_path).convert("RGB")
            
            
            anno_path = osp.join(self.root, "annotations", img_name)
            anno = loadmat(anno_path)
            box_key = "box_new"
            if box_key not in anno.keys():
                box_key = "anno_file"
            if box_key not in anno.keys():
                box_key = "anno_previous"

            rois = anno[box_key][:, 1:]
            ids = anno[box_key][:, 0]
            rois = np.clip(rois, 0, None)  # several coordinates are negative

            assert len(rois) == len(ids)
            result = [self.start + i for i in range(len(rois))]
            self.start = result[-1] + 1
            rois[:, 2:] += rois[:, :2]
            indices = np.where(ids == -2)[0]
            for i, idx in enumerate(indices):
                ids[idx] = self.unlabel_start
                self.unlabel_start += 1
            # ids[ids == -2] = 5555  # assign pid = 5555 for unlabeled people
            # self.label2instance[]
            
            
            # cropped_paths = []
            # for i, (roi, res_id, label) in enumerate(zip(rois, result, ids)):
            #     if label == 5555:
            #         self.unlabel_count+=1
            #         self.label2instance[self.unlabel_count] = [res_id]
            #     elif label in self.label2instance:
            #         self.label2instance[int(label)].append(res_id)
            #     else:
            #         self.label2instance[int(label)] = [res_id]
                # x1, y1, x2, y2 = roi
                # # Ensure coordinates are integers
                # x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # # Optional: Validate ROI coordinates within image dimensions
                # width, height = image.size
                # x1 = max(0, min(x1, width - 1))
                # y1 = max(0, min(y1, height - 1))
                # x2 = max(x1 + 1, min(x2, width))
                # y2 = max(y1 + 1, min(y2, height))

                # # Crop the image using the ROI
                # cropped_img = image.crop((x1, y1, x2, y2))

                # # Define the cropped image name based on the unique instance ID
                # cropped_name = f"{res_id}.jpg"  # You can retain the original extension if needed
                # cropped_path = osp.join(cropped_dir, cropped_name)

                # # Save the cropped image only if it doesn't already exist to avoid redundancy
                # if not osp.exists(cropped_path):
                #     try:
                #         cropped_img.save(cropped_path)
                #     except IOError as e:
                #         print(f"Error saving cropped image {cropped_path}: {e}")
                #         continue  # Skip saving this cropped image if there's an error

                # # Append the path of the cropped image to the list
                # cropped_paths.append(cropped_path)
            
            
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": rois.astype(np.int32),
                    # (training pids) 1, 2,..., 478, 480, 481, 482, 483, 932, 5555
                    "pids": ids.astype(np.int32),
                    "cam_id": self._get_cam_id(img_name),
                    "instances_id": result
                }
            )
        return annotations
