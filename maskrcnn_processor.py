#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 08:35:40 2020
 _______  __   __  _______
|       ||  | |  ||       |
|  _____||  |_|  ||  _____|
| |_____ |       || |_____
|_____  ||       ||_____  |
 _____| | |     |  _____| |
|_______|  |___|  |_______|

MASK RCNN PROCESSOR
@author: jakub
"""
import os
import multiprocessing as mp
import os.path as pt
from queue import Empty

class MaskRCNNProc:
    def __init__(self, img_Q, mrcnn_Q):
        self.stopped = mp.Event()
        self.MODEL_PATH = pt.join('models', 'mask_rcnn_covid_0034.h5')
        self.p = mp.Process(target=self._loop, args=(img_Q, mrcnn_Q,))
        self.p.start()
    def _loop(self, img_Q, mrcnn_Q):
        import tensorflow as tf
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        from maskrcnn_config import InferenceConfig
        from mrcnn import model as modellib

        inference_config = InferenceConfig()
        # Recreate the model in inference mode
        model = modellib.MaskRCNN(mode="inference",
                                  config=inference_config,
                                  model_dir='models')
        model.load_weights(self.MODEL_PATH, by_name=True)
        while not self.stopped.is_set():
            try:
                img = img_Q.get(timeout=1)
            except Empty:
                continue
            results = model.detect([img], verbose=1)
            mrcnn_Q.put(results)
    def stop(self):
        self.stopped.set()
        self.p.join()
        self.p.terminate()

if __name__ == "__main__":
    import cv2 as cv
    test_images_dir  = pt.join('test_data', 'images')
    test_images = os.listdir(test_images_dir)
    img_Q = mp.Queue()
    mrcnn_Q = mp.Queue()
    mrcnn = MaskRCNNProc(img_Q, mrcnn_Q)
    N = len(test_images)
    for img_fn in test_images:
        img_pth = pt.join(test_images_dir, img_fn)
        img = cv.imread(img_pth)
        img_Q.put(img)
    total_results = []
    for i in range(N):
        total_results.append(mrcnn_Q.get())
    mrcnn.stop()
