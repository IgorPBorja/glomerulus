import sys
import os
import numpy as np
import skimage
import skimage.io, skimage.filters, skimage.exposure, skimage.util
import scipy.ndimage
from skimage.restoration.uft import laplacian
import shutil
from tqdm import tqdm
import argparse

ndim = 3 ## images (H, W, C) ==> 3-tensors, ndim=3
sigma = 5.0
truncate = 4.0
gammas = [0.1, 0.6, 2.5]
consts = [1.0, 1.0, 1.0]
hist_bins = 256
ksize = 3
laplace_const = 10.0
allowed_extensions = [".jpg", ".png", ".jpeg", ".JPG"]

def init_parser():
    parser = argparse.ArgumentParser(description="Apply image transformation to given dataset")
    parser.add_argument("dataset_path", type=str)
    parser.add_argument("--laplace", action="store_true")
    parser.add_argument("--gamma", action="store_true")
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--hist", action="store_true")
    return parser


def apply_laplace_filter_channelwise(img: np.ndarray, ksize: int):
    ## See https://github.com/scikit-image/scikit-image/blob/v0.21.0/skimage/filters/edges.py#L666-L701
    ## for original implementation
    ## shape = (k, k, k), where k = kernel_size
    out = np.zeros_like(img)
    for i in range(img.shape[-1]):
        out[:, :, i] = np.clip(skimage.filters.laplace(img[:, :, i], ksize=ksize), 0.0, 1.0)
    laplace_non_scaled = skimage.color.rgb2gray(out)
    return np.clip(laplace_const * laplace_non_scaled, 0.0, 1.0)

def apply_hist_equalize_channelwise(img: np.ndarray, nbins: int = 256):
    out = np.zeros_like(img)
    for i in range(img.shape[-1]):
        out[:, :, i] = skimage.exposure.equalize_hist(img[:, :, i], nbins=nbins)
    return out


def init_transforms(dataset_path) -> list:
    img_transforms = [
        (
            "gaussian",
            dataset_path + "_gaussian", 
            lambda float_img: skimage.filters.gaussian(float_img, sigma, truncate=truncate, mode="constant", cval=0, channel_axis=-1), 
            f"Gaussian filter with maximum std of sigma={sigma}, truncating after {truncate} standard deviations"
        ),
        *(
            (
                "gamma",
                dataset_path + f"_gamma{n+1}", 
                lambda float_img: skimage.exposure.adjust_gamma(float_img, gammas[n], gain=consts[n]), 
                f"Gamma correction with gamma={gammas[n]} and scaling factor {consts[n]}"
            ) for n in range(len(gammas))
        ),
        # convolution with laplace filter (channelwise)
        ( 
            "laplace",
            dataset_path + f"_laplace",
            lambda float_img: apply_laplace_filter_channelwise(float_img, ksize=ksize), 
            f"Laplace operator with kernel K of size 3"
         ),
        (
                "hist",
                dataset_path + f"_hist",
                lambda float_img: apply_hist_equalize_channelwise(float_img, nbins=hist_bins),
                f"Histogram equalization with {hist_bins} disjoint, equal-sized bins"
        ) 
    ]
    return img_transforms

def main():
    parser = init_parser()
    args = parser.parse_args()

    img_transforms = init_transforms(args.dataset_path)

    for (name, new_root, T, desc) in img_transforms:
        # copy directory structure
        # see https://stackoverflow.com/questions/15663695/shutil-copytree-without-files

        # guard clause
        if (name == "gaussian" and not args.gaussian) or (name == "gamma" and not args.gamma) or (name == "laplace" and not args.laplace) or (name == "hist" and not args.hist):
            continue

        def ignore_files(dir, files):
            return [f for f in files if os.path.isfile(os.path.join(dir, f))]

        shutil.copytree(args.dataset_path, new_root, ignore=ignore_files,
                        dirs_exist_ok=True) ## the last option implies "overwrite every time"

        for cwd, cwd_subdirs, files in os.walk(args.dataset_path):
            ## use tqdm progress bar if in leaf directory (no other subdirectories), which means the images are being created
            if cwd_subdirs == []:
                progress_bar_iterable = tqdm(files)
                print(f"Creating folder {cwd.replace(args.dataset_path, new_root)}")
            else:
                progress_bar_iterable = files
            
            for file in progress_bar_iterable: ## use progress bar
                if not any([file.endswith(ext) for ext in allowed_extensions]):
                    continue
                filepath = os.path.join(cwd, file)
                img = skimage.io.imread(filepath)
                img_float = skimage.util.img_as_float(img)
            
                new_img_float = np.clip(T(img_float), 0.0, 1.0) ## avoid precision errors like 1.0000002
                new_img_ubyte = skimage.util.img_as_ubyte(new_img_float)
                new_filepath = filepath.replace(args.dataset_path, new_root, 1)
                skimage.io.imsave(new_filepath, new_img_ubyte, check_contrast=False)
        
        print(f"Finalized transformation '{desc}'")

main()
