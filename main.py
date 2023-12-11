# import the necessary packages
import numpy as np
import os
import argparse
import cv2
import json
import pathlib
import  logging

from tqdm import tqdm
from utils import convert, visualize 
from utils.labelme import COCOInstance2Labelme
from skimage import measure
from pycocotools import mask as coco_mask

def save_overlap_seg(info:dict, curr_img:np.ndarray,
                     save_visulize_result:str, polygons:list):
    img_svae_path = os.path.join(save_visulize_result,source_img_name)
    curr_img = visualize.polygon2overlay_mask(curr_img, polygons, info['bbox'])
    cv2.imwrite(img_svae_path, curr_img)
    return curr_img
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--json_file", default="coco_instances_results.json" ,
                        help="coco evaluation json file", type=str)
    parser.add_argument("-t", "--threshold", default=0.5 ,
                        help="output results above the threshold", type=float)
    parser.add_argument("-s", "--source_img_dir", default="./image_data",
                        help="source image directory", type=str)
    parser.add_argument("-f", "--format", default=".jpg",
                        help="source image format", type=str)
    parser.add_argument("-tol", "--tolerance", default=5,
                        help="0-100, lower for lossness mask accuracy", type=int)
    parser.add_argument("-v", "--save_visulize_result", default="",
                        help="save visulize result", type=str)
    parser.add_argument("-sj", "--save_json_path", default="./result/labelme/",
                        help="save json file with labelme format", type=str)
    args = parser.parse_args()

    with open(args.json_file) as f:
        coco_format = json.load(f)
        ins_num = len(coco_format)
        logging.basicConfig(level=logging.INFO)
        logging.info(f'Found {ins_num} instances.')
    
    if args.save_visulize_result != "":
        pathlib.Path(args.save_visulize_result).mkdir(parents=True, exist_ok=True)
    if args.save_json_path != "":
        pathlib.Path(args.save_json_path).mkdir(parents=True, exist_ok=True)
        c2l = COCOInstance2Labelme(args.source_img_dir, [args.format])

    curr_img_path = ''
    
    progress = tqdm(ins_num)
    for i, info in enumerate(coco_format):
        if info['score'] >= args.threshold:
            coco_rle_string = info['segmentation']
            seg_mask = np.array(coco_mask.decode(coco_rle_string), dtype=np.float32)
            polygons = convert.binary_mask_to_polygon(seg_mask, args.tolerance)
            polygons = convert.pair_coord(polygons, info['bbox'])
            
            if args.save_json_path != "":
                c2l.input_gt(info["image_id"], info["category_id"], polygons.tolist())
                c2l.save_json(args.save_json_path, info["image_id"]+'.json')
            
            if curr_img_path == info['image_id']:
                pass
            else:
                curr_img_path = info['image_id']
                source_img_name = info['image_id']+args.format
                curr_img = cv2.imread(f'{args.source_img_dir}/{source_img_name}')
            if args.save_visulize_result != "":
                curr_img = save_overlap_seg(info, curr_img,args.save_visulize_result, polygons)
        
        progress.update(1)