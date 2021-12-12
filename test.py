#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mask_rcnn_pytorch 
@File    ：test.py
@Author  ：zhuofalin
@Date    ：2021/11/24 21:19 
'''
import cv2, os

# img = cv2.imread('inference/test_image/FudanPed00002.png')
# print(img.shape)
# cv2.imshow('img', img)
# cv2.waitKey()
# (414, 455, 3)

weight_save_path = 'weights'
weight_save_name = 'mask_R_CNN_model.pth'
print(os.path.join(weight_save_path, weight_save_name))