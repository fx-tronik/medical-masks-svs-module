#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:46:47 2020
 _______  __   __  _______ 
|       ||  | |  ||       |
|  _____||  |_|  ||  _____|
| |_____ |       || |_____ 
|_____  ||       ||_____  |
 _____| | |     |  _____| |
|_______|  |___|  |_______|

Mask use validator
@author: jakub
"""
import os
import multiprocessing as mp
import numpy as np
import os.path as pt
from queue import Empty
from maskrcnn_processor import MaskRCNNProc
from hpe_processor import HPEProc
from hpe_utils import whole_human_bb

def point_in_bb(tlbr, pp):
    #check if point is in bb
    tlx, tly, brx, bry = tlbr
    px, py = pp
    res = (px >= tlx) and (px <= brx) and (py >= tly) and (py <= bry)
    return res

def get_nose_point(human, w, h):
    nid = 0 #nose id in network
    if not (nid in human.body_parts.keys()):
        return None
    else:
        nx, ny = w * human.body_parts[nid].x, h * human.body_parts[nid].y
        return nx, ny
    
def get_contours(med_ms):
    med_m = med_ms[0]
    med_masks = med_m['masks']    
    h, w, N = med_masks.shape
    contours  = []
    for i in range(N):
        mask = med_masks[:, :, i].astype(np.uint8)
        contour, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        contour = contour[0]
        contours.append(contour)
    return contours, h, w
    

def match_mm(human, contours, w, h):
    #rois = med_m['rois']
    #N = len(rois)
    npt = get_nose_point(human, w, h)
    if npt is None:
        return None
    for i, cnt in enumerate(contours):
        bx,by,bw,bh = cv.boundingRect(cnt)
        dist = cv.pointPolygonTest(cnt, npt, True)
        if dist > (- bh): #it's a match
            return i, dist 
    else:
        return None    


        
        
if __name__ == "__main__":
    import cv2 as cv
    import matplotlib.pyplot as plt
    test_images_dir  = pt.join('test_data', 'images')
    output_images_dir  = pt.join('test_data', 'output')
    test_images = os.listdir(test_images_dir)
    img_h_Q = mp.Queue()
    hpe_Q = mp.Queue()
    hpe = HPEProc(img_h_Q, hpe_Q)
    img_m_Q = mp.Queue()
    mrcnn_Q = mp.Queue()
    mrcnn = MaskRCNNProc(img_m_Q, mrcnn_Q)
    N = len(test_images)
    images = []
    for img_fn in test_images:
        img_pth = pt.join(test_images_dir, img_fn)
        img = cv.imread(img_pth)
        img_h_Q.put(img)
        img_m_Q.put(img)
        images.append(img)
    total_h_results = []
    total_m_results = []
    for i in range(N):
        total_h_results.append(hpe_Q.get())
        total_m_results.append(mrcnn_Q.get())
    hpe.stop()
    mrcnn.stop()
    i = 0
    for humans, masks, image in zip(total_h_results, total_m_results, images):
        print('Next image')
        h, w, _ = image.shape
        vis = image.copy()
        cnts = get_contours(masks)
        if not(cnts is None): #some medical masks found
            contours, h, w = cnts
            for human in humans:
                match = match_mm(human, contours, w, h)
                if not(match is None):
                    med_m_id, dist = match                    
                    valid = dist > 0
                    roi = masks[0]['rois'][med_m_id]
                    color = (0,255,0) if valid else (255, 0, 0)
                    vis = cv.rectangle(vis, (roi[1], roi[0]), (roi[3], roi[2]), color, w//250)
                
                nose_pt = get_nose_point(human, w, h)
                if not(nose_pt is None):
                    nx, ny = nose_pt
                    nix, niy = np.round([nx, ny]).astype(np.int32)
                    vis = cv.circle(vis, (nix, niy), w // 100, (0,0,255), -1)
        #plt.imshow(vis[:, :, ::-1])
        #plt.show()
        cv.imwrite(pt.join(output_images_dir, test_images[i]), vis)
        i+=1


# BOX
# =============================================================================
#             box = human.get_face_box(w, h, mode=1)
#             box = whole_human_bb(human, w, h)
#             if not box:
#                 continue
#             draw face box
#             p1 = tuple([face_box['x'] - face_box['w'] // 2, face_box['y'] - face_box['h'] // 2])
#             p2 = tuple([face_box['x'] + face_box['w'] // 2, face_box['y'] + face_box['h'] // 2])
#             tlx, tly, bw, bh = box['x'] - box['w'] // 2, box['y'] - box['h'] // 2, box['w'], box['h'] #for face box
#             tlx, tly, bw, bh = box # for whole human bb
#             vis = cv.rectangle(vis, (tlx, tly), (tlx+bw, tly+bh), (0,255,0), 4)
# =============================================================================