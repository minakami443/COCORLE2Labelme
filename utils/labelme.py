import json
import os
import cv2

import numpy as np
from typing import List

class COCOInstance2Labelme():
    def __init__(self, image_dir, image_format:List[str]=['.jpg']):
        self.image_dir = image_dir
        self.image_format = image_format
        self.image_list = os.listdir(self.image_dir)
        self.label_dict =self._labelme()

    def _labelme(self)->dict:
        label_dict = {}
        
        for fn in self.image_list:
            subn = '.'+fn.split('.')[-1]
            if subn in self.image_format:
                fp = os.path.join(self.image_dir, fn)
                img = cv2.imread(fp)
                height, width, channel = img.shape
                img_id = os.path.splitext(fn)[0]
                label_dict[img_id] = {
                    "version": "4.5.6",
                    "flags": {},
                    "shapes": [],
                    "imagePath": fn,
                    "imageData": None,
                    "imageHeight": height,
                    "imageWidth": width
                    }
        return label_dict
    
    def input_gt(self, img_id:str, cls:str, points:list):
        gt_dict = {
            "label": cls,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
            }
        self.label_dict[img_id]["shapes"].append(gt_dict)

    def save_json(self, save_path:str = './labelme/', filename:str = 'instance.json'):
        sfn_path = os.path.join(save_path, filename)
        with open(sfn_path, "w+") as f:
            img_id =os.path.splitext(filename)[0]
            json.dump(self.label_dict[img_id], f)

                
