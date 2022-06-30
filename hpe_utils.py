#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 08:49:39 2020

@author: jakub
"""
import cv2
import numpy as np
import colorsys

#Human processing functions
CocoPairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11),
    (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17), (2, 16), (5, 17)
]
def body_parts_distance(part, other_part, w, h):
    #normalized distances:
    dx = np.abs(part.x - other_part.x)
    dy = np.abs(part.y - other_part.y)
    #distance in pixels
    rdx, rdy = w * dx, h * dy
    return (rdx ** 2 + rdy **2) ** 0.5    

def get_base_dim(human, w, h, k = 0.5):
    #defin bbox size by computing length of limbs (ankle->knee, wrist->elbow)
    #bbox will have a size of k * mean limb length

    #human parts in coco pairs:
    parts = [2, 3, 4, 5, #arms
             7, 8, 10, 11] # legs
    selected_pairs = [CocoPairs[i] for i in parts]
    limb_lengths = []
    for pair in  selected_pairs:
        part_u, part_v = pair
        if part_u in human.body_parts.keys() and part_v in human.body_parts.keys():
            leng  = body_parts_distance(human.body_parts[part_u],
                                        human.body_parts[part_v], w, h)
            limb_lengths.append(leng)
    if len(limb_lengths):
        return int(k * np.mean(limb_lengths) + 0.5)
    else:
        return None

def whole_human_bb(human, w, h, k = 0):
    base_size = get_base_dim(human, w, h)
    if base_size is None:
        return None
    margin = k * base_size
    #find max, min x, y and add margin k * base_dim
    xs = [w * part.x for part in human.body_parts.values()]
    ys = [h * part.y for part in human.body_parts.values()]
    tlx = max([min(xs) - margin, 0])
    tly = max([min(ys) - margin, 0])
    brx = min([max(xs) + margin, w])
    bry = min([max(ys) + margin, h])
    bbx = [tlx, tly, brx - tlx, bry - tly]
    bbx = [int(x) for x in bbx]
    return bbx

def create_unique_color_float(tag, hue_step=0.41):
    """Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (float, float, float)
        RGB color code in range [0, 1]

    """
    h, v = (tag * hue_step) % 1, 1. - (int(tag * hue_step) % 4) / 5.
    r, g, b = colorsys.hsv_to_rgb(h, 1., v)
    return r, g, b

def create_unique_color_uchar(tag, hue_step=0.41):
    
    """ from deep sort visualization
    Create a unique RGB color code for a given track id (tag).

    The color code is generated in HSV color space by moving along the
    hue angle and gradually changing the saturation.

    Parameters
    ----------
    tag : int
        The unique target identifying tag.
    hue_step : float
        Difference between two neighboring color codes in HSV space (more
        specifically, the distance in hue channel).

    Returns
    -------
    (int, int, int)
        RGB color code in range [0, 255]

    """
    r, g, b = create_unique_color_float(tag, hue_step)
    return [int(255*r), int(255*g), int(255*b)]

def draw_tracker(tracker, image):
    thickness = 2
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        color = create_unique_color_uchar(track.track_id)
        bbox = track.to_tlbr().astype(np.int)
        pt1, pt2 = tuple(bbox[:2]), tuple(bbox[2:4])
        image = cv2.rectangle(image, pt1, pt2, color, thickness)
        
        meanx, meany = int(np.mean([pt1[0], pt2[0]])), int(np.mean([pt1[1], pt2[1]]))
        text_id = track.track_id
        font = cv2.FONT_HERSHEY_DUPLEX
        image = cv2.putText(image, str(text_id), (meanx, meany), font, 
                            1, color, 2, cv2.LINE_AA)
    return image



    