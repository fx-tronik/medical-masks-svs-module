#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:13:45 2020
 _______  __   __  _______
|       ||  | |  ||       |
|  _____||  |_|  ||  _____|
| |_____ |       || |_____
|_____  ||       ||_____  |
 _____| | |     |  _____| |
|_______|  |___|  |_______|

Human pose estimator processor
@author: jakub
"""
import os
import multiprocessing as mp
import os.path as pt
from queue import Empty

class HPEProc:
    def __init__(self, img_Q, hpe_Q):
        self.stopped = mp.Event()
        self.MODEL_PATH = pt.join('models', 'mask_rcnn_covid_0027.h5')
        self.p = mp.Process(target=self._loop, args=(img_Q, hpe_Q,))
        self.p.start()
    def _loop(self, img_Q, mrcnn_Q):
        import tensorflow as tf
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.3)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
        from tf_pose.estimator import TfPoseEstimator
        from tf_pose.networks import get_graph_path, model_wh
        model = pt.join('models', 'hpe_cmu_graph_opt.pb')
        resolution = '1280x720'
        w, h = model_wh(resolution)
        e = TfPoseEstimator(model, target_size=(w, h))

        while not self.stopped.is_set():
            try:
                img = img_Q.get(timeout=1)
            except Empty:
                continue
            results = e.inference(img, upsample_size = 8.0)
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
    hpe_Q = mp.Queue()
    hpe = HPEProc(img_Q, hpe_Q)
    N = len(test_images)
    for img_fn in test_images:
        img_pth = pt.join(test_images_dir, img_fn)
        img = cv.imread(img_pth)
        img_Q.put(img)
    total_results = []
    for i in range(N):
        total_results.append(hpe_Q.get())
    hpe.stop()
