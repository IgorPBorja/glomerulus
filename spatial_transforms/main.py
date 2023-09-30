import skimage
import numpy as np
import os
import matplotlib.pyplot as plt
import transforms
import time

DATASET_PATH = "Glomerulus"
MAX_ITER = 1
c = 1.5
gamma = 0.7
N = 10
sigma = 5.0

def switch_T(img: np.ndarray, t: str):
    if t == "gamma":
        return transforms.gamma_correction(img, gamma=gamma, c=c)
    elif t == "gauss":
        return transforms.GaussianFilter.apply_gaussian_filter(img, N=N, sigma=sigma) 

if __name__ == "__main__":
    test_subpath = "Normal/treino/AZAN"
    complete_path = os.path.join(DATASET_PATH, test_subpath)
    
    print(f"Searching on path {complete_path}")
    files = os.listdir(complete_path)
    print(f"Found {len(files)} images")
    
    i: int = 0
    for f in files:
        prepended_f = os.path.join(complete_path, f)
        img = skimage.img_as_float(skimage.io.imread(prepended_f))
        
        t0 = time.time()
        my_gaussian = transforms.GaussianFilter.apply_gaussian_filter(img, N=N, sigma=sigma) 
        t1 = time.time()
        print(f"My gaussian took {t1 - t0} seconds")
        t0 = time.time()
        skimage_gaussian = skimage.filters.gaussian(img, sigma=sigma,
                                                    mode="constant",
                                                    cval=0.0,
                                                    channel_axis=-1)
        t1 = time.time()
        print(f"Skimage's gaussian took {t1 - t0} seconds")
        
        fig, axis = plt.subplots(2, 2)
        
        for i, arr, title in zip(
            range(2 * 2), 
            (img, my_gaussian, skimage_gaussian),
            ("Original image",
             f"My gaussian transform (N={N}, sigma={sigma})",
             f"Skimage's gaussian transform (sigma={sigma})")
        ):
            axis[i // 2, i % 2].imshow(arr)
            axis[i // 2, i % 2].axis("off")
            axis[i // 2, i % 2].set_title(title)
        
        plt.show()
        
        i += 1
        if i >= MAX_ITER:
            break