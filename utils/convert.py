import numpy as np
from skimage import measure

def close_contour(contour:np.ndarray):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(bin_mask:np.ndarray, tolerance:int=0):
    polygons = []

    pad_bin_mask = np.pad(bin_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(pad_bin_mask, 0.5)
    for contour in contours:
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) <3:
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [0 if i < 0 else i for i in segmentation]
        polygons.append(segmentation)
    return polygons

def pair_coord(coords:np.ndarray, bbox:list)->np.ndarray:
    points = None
    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[0]+bbox[2]), int(bbox[1]+bbox[3])
    for coord in coords:
        for i in range(0,len(coord),2):
            if coord[i] <= xmin:
                px = xmin
            elif xmin < coord[i] < xmax:
                px = coord[i]
            elif coord[i] >= xmax:
                px = xmax
            if coord[i+1] <= ymin:
                py = ymin
            elif ymin < coord[i+1] < ymax:
                py = coord[i+1]
            elif coord[i+1] >= ymax:
                py = ymax
            point = np.array([px, py], dtype=np.int32).reshape(1,2)
            if (points is None):
                points = point
            else:
                points = np.concatenate([points, point], axis=0)
    return points
