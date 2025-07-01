import numpy 
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image, ImageOps

class MyImage:
    def change_size(image, target_size, pad_value=0):
        
        target_h, target_w = target_size
        img_h, img_w = image.shape[:2]
        image = cv2.resize(image, (target_w * img_w // img_h, target_h))
        img_h, img_w = image.shape[:2]

        # crop
        start_x = max(0, (img_w - target_w) // 2)
        start_y = max(0, (img_h - target_h) // 2)
        end_x = start_x + min(target_w, img_w)
        end_y = start_y + min(target_h, img_h)

        cropped = image[start_y:end_y, start_x:end_x]

        # pad
        pad_h = max(0, target_h - cropped.shape[0])
        pad_w = max(0, target_w - cropped.shape[1])
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        if image.ndim == 3:
            padding = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
        else:
            padding = ((pad_top, pad_bottom), (pad_left, pad_right))

        padded = np.pad(cropped, padding, mode='constant', constant_values=pad_value)

        return padded


class MyTyping:
    sort_dict = lambda x: dict(reversed(sorted(x.items(), key = lambda item: item[1])))