import numpy as np
import cv2

from typing import List

def poly2mask(polygon:List[int], width:int, height:int, visualize:bool =False):
    obj = np.array([polygon], dtype=np.int32)
    if visualize:
        mask = np.zeros((height, width, 3), dtype=np.int32)
        cv2.fillPoly(mask, obj, [1,1,1])
        mask*255
    else:
        mask = np.zeros((height, width), dtype=np.int32)
        cv2.fillPoly(mask, obj, 1)
    return mask

def polygon2overlay_mask(source:np.ndarray, polygon:List[List[int]], bbox:list or False = False):
    color = np.random.randint(0, 255, size=(3, )).tolist()
    height = source.shape[0]
    width = source.shape[1]

    mask = poly2mask(polygon, width, height, True)
    cv2.polylines(source, [polygon], True, color,1)
    color_mask = mask*color
    source = 0.5*color_mask+source
    
    if bbox:
        xminymin = np.array([bbox[0], bbox[1]], dtype=np.int32)
        xmaxymax = np.array([bbox[0]+bbox[2], bbox[1]+bbox[3]], dtype=np.int32)
        source =cv2.rectangle(source, xminymin, xmaxymax, color, 2)
    return source