#!/usr/bin/env python3

import skimage
from skimage import io
import numpy as np

DATASET_PATH = "Glomerulus"

def gamma_correction(img: np.ndarray, gamma: float=1.0, c: float=1.0):
    """
    Applies gamma correction to image ndarray (3-tensor) with scaling constant c
    and exponent gamma.
    Assumes normalized image (in [0,1]) and clips the resulting values to [0,1] also
    """
    assert(len(img.shape) == 3) # (C, H, W) or (H, W, C)
    
    return np.clip(c * (img ** gamma), 0.0, 1.0) # element-wise operations

class GaussianFilter:
    @staticmethod 
    def gaussian(x: float, sigma: float):
        return (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(- x**2 / (2.0 * sigma**2))
    
    @staticmethod
    def init_kernel(N: int, sigma: float) -> np.ndarray:
        gaussian_arr = GaussianFilter.gaussian(np.arange(start=-N, stop=N + 1).astype(np.float64), sigma)
        
        # return tensor outer product
        return np.outer(gaussian_arr, gaussian_arr)
    
    @staticmethod
    def apply_gaussian_filter(img: np.ndarray, N: int, sigma: float) -> np.ndarray:
        output_img = np.zeros_like(img)
        filter = GaussianFilter.init_kernel(N, sigma) # (2N + 1) times (2N + 1)
        #print(f"Original img shape: {img.shape}") ## TODO REMOVE
        for c in range(img.shape[2]):
            #print(f"_: {img[:, :, c].shape}") ## TODO REMOVE
            padded_channel = np.pad(img[:, :, c], ((N, N), (N, N)), 
                                    constant_values=((0,0),(0,0)),
                                    mode="constant")
            #print(f"Shape of total padded channel {padded_channel.shape}") ## TODO REMOVE
            for i in range(N, img.shape[0] + N):
                for j in range(N, img.shape[1] + N):
                    #print(f"Shape of patch: {padded_channel[i - N: i + N + 1, j - N: j + N + 1].shape} \t Shape of filter: {filter.shape}") ## TODO REMOVE
                    conv = padded_channel[i - N: i + N + 1, j - N: j + N + 1] * filter ## element-wise product
                    output_img[i - N][j - N][c] = np.sum(conv)
        return output_img
                    
    

        
        
        
        
    
     
