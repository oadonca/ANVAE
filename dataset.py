# -*- coding: utf-8 -*-
"""
Created on Sun Sep 13 19:10:38 2020

@author: Octavian
"""
import os
import time
from datetime import datetime
import pandas
import numpy as np
import cv2

_RNG_SEED = None

class dataset(object):
    def __init__(self, data_path, batch_size):
        self.image_id = 0
        self.data_path = data_path
        self.batch_size = batch_size
        self.epochs_completed = 0
        self.img_dim = (32, 32)
        
        self.setup(epoch_val=0, batch_size=batch_size)
        self.load_files(data_path)

    def load_files(self, data_path):
        
        self.data = np.array(pandas.read_csv(data_path), dtype=np.uint8)
        
        self.labels = self.data[:,:1]
        self.data = self.data[:,1:]
        
        self.n_samples = self.data.shape[0]
        rows = 28
        cols = 28
        self.data = np.reshape(self.data, (self.n_samples, rows, cols, 1))
        self.temp_data = np.empty([self.n_samples, 32, 32, 1])
        
        for i in range(self.n_samples):
            h, w, c  = self.data[i].shape
            # self.img_dim  = (self.img_dim[0], self.img_dim[1], c)
            if h > w:
                top_h = int((h - w) / 2)
                self.data[i] = self.data[i][top_h:top_h + w, :, :]
            else:
                left_w = int((w - h) / 2)
                self.data[i] = self.data[i][:, left_w:left_w + h, :]
    
            a = self.img_dim[0]
            b = self.img_dim[1]
            
            image = cv2.resize(self.data[i], self.img_dim, interpolation=cv2.INTER_LINEAR)

            self.temp_data[i] = np.reshape(image, (a, b, -1))
            self.temp_data[i] = self.temp_data[i] / 255.
            
        self.data = self.temp_data

        self.suffle_files()
        
    def suffle_files(self):
        idxs = np.arange(self.size())

        self.rng.shuffle(idxs)
        
        self.data = self.data[idxs]
        self.labels = self.labels[idxs]

    def size(self):
        return self.data.shape[0]

    def next_batch(self):
        assert self.batch_size <= self.size(), \
          "batch_size {} cannot be larger than data size {}".\
           format(self.batch_size, self.size())
        start = self.image_id
        self.image_id += self.batch_size
        end = self.image_id
        batch_data = self.data[start:end]
        batch_labels = self.labels[start:end]

        if self.image_id + self.batch_size > self.size():
            self.epochs_completed += 1
            self.image_id = 0
            self.suffle_files()
        return [batch_data, batch_labels]

    def setup(self, epoch_val, batch_size, **kwargs):
        self.reset_epochs_completed()
        self.reset_state()
        
    def reset_epoch(self):
        self.epochs_completed = 0

    def reset_epochs_completed(self):
        self.epochs_completed = 0

    def reset_state(self):
        self.rng = self.get_rng()
        
    def get_rng(obj=None):
        """
        This function is copied from `tensorpack
        <https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/utils.py>`__.
        Get a good RNG seeded with time, pid and the object.
        Args:
            obj: some object to use to generate random seed.
        Returns:
            np.random.RandomState: the RNG.
        """
        seed = (id(obj) + os.getpid() +
                int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295
        if _RNG_SEED is not None:
            seed = _RNG_SEED
        return np.random.RandomState(seed)