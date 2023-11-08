import skimage.io
from dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
import sys 
import os.path
import config
import random

def extract_img_glcm(
        glcm_featuremap: np.ndarray, 
        contrast_matrix: np.ndarray,
        n: int, 
        i: int = 0, 
        j: int = 0) -> tuple[np.ndarray[any, np.float64], np.float64]:
    glcm_featuremap = glcm_featuremap.reshape((-1, config.MAXL + 1, config.MAXL + 1, len(config.DISTANCES), len(config.ANGLES)))
    ## glcm matrix is of type np.float64
    ## because of normed
    ## get item (i, j) (distance i paired with angle j)
    assert(0 <= i and i < glcm_featuremap.shape[-2])
    assert(0 <= j and j < glcm_featuremap.shape[-1])
    img_as_uint32 = glcm_featuremap[n, :, :, i, j]
    img_as_float64 = img_as_uint32.astype(np.float64)
    
    contrast = contrast_matrix[n, i, j]
    return img_as_float64, contrast
 
if __name__ == '__main__':
    num_imgs = 3
    DS_PATH = sys.argv[1]
    if (len(sys.argv) > 2):
        CUTOFF = float(sys.argv[2])
    else:
        CUTOFF = float('inf')
    contrast_path = os.path.basename(DS_PATH + "_contrast.npy")
    contrast_matrix = np.load(contrast_path).reshape(-1, len(config.DISTANCES), len(config.ANGLES))
    D : Dataset = Dataset(DS_PATH)

    info = {
            "LBP": [os.path.basename(DS_PATH) + "_LBP.jpeg", 
                     "Image (left), image as graylevel (middle) \n and image representation of its local binary patterns (LBP) (right)",
                    lambda featuremap, i: featuremap[i, :].reshape(config.SHAPE) ],
            "GLCM": [os.path.basename(DS_PATH) + "_GLCM.jpeg",
                     "Image (left), image as graylevel (middle) \n and image representation of its graylevel co-ocurrence matrix (GLCM) (right)", 
                     lambda featuremap, i: extract_img_glcm(featuremap,contrast_matrix, i)],
            "sobel": [os.path.basename(DS_PATH) + "_sobel.jpeg", 
                     "Image (left), image as graylevel (middle) \n and image representation of an application of a Sobel filter on it (right)",
                      lambda featuremap, i: featuremap[i, :].reshape(config.SHAPE)],
            }

    ## Plotting

    ## Change font of axis title and x/y-labels 
    plt.rcParams["figure.figsize"] = (14, 12)
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 12
    plt.subplots_adjust(left=None, bottom=None, right=None, top=2.0, wspace=None, hspace=0.3)
    #  plt.tight_layout()

    for key, (save_file, title, extractor) in info.items():
        featuremap = np.load(os.path.basename(DS_PATH) + f"_{key}.npy")
        ## Use different random images for each transformation
        idx = [random.randint(0, min(len(D), CUTOFF)) for _ in range(num_imgs)]

        fig, ax = plt.subplots(num_imgs, 3, constrained_layout=True)
        #  fig.tight_layout()
        #  fig.suptitle(title, fontsize=18)
        for ax_cnt, i in enumerate(idx):
            main_img = skimage.io.imread(D[i])
            main_img = skimage.transform.resize(main_img, config.SHAPE)   
            main_img_as_gray = skimage.util.img_as_ubyte(skimage.color.rgb2gray(main_img)) # to gray_level
            if (key == "GLCM"):
                feature_img, contrast = extractor(featuremap, i) 
            else:
                feature_img = extractor(featuremap, i) 
            ax[ax_cnt][0].imshow(main_img, aspect="auto")
            # Hide ticks (numbering) in axis
            ax[ax_cnt][0].set_xticklabels([])
            ax[ax_cnt][0].set_yticklabels([])
            ax[ax_cnt][0].set_xlabel("Original image (RGB)")
            ax[ax_cnt][1].imshow(main_img_as_gray, cmap='gray', aspect="auto")
            ax[ax_cnt][1].set_xlabel("Original image (transformed to grayscale)")
            # Hide ticks (numbering) in axis
            ax[ax_cnt][1].set_xticklabels([])
            ax[ax_cnt][1].set_yticklabels([])
            ax[ax_cnt][2].imshow(feature_img, cmap="gray", aspect="auto")
            # Hide ticks (numbering) in axis
            ax[ax_cnt][2].set_xticklabels([])
            ax[ax_cnt][2].set_yticklabels([])
            if (key == "GLCM"):
                ax[ax_cnt][2].set_xlabel(f"Feature: {key} \n Constrast: {contrast:.4f}")
            else:
                ax[ax_cnt][2].set_xlabel(f"Feature: {key}")
        plt.savefig(save_file)
        print(f"Plotted feature {key}")
        del featuremap # force memory saving

    ## Now, plot histograms
    ## Choosing only images from original dataset  
    _compressed = np.load("LBP_histograms.npz")
    lbp_histograms = _compressed['lbp_histograms']
    lbp_bin_edges = _compressed['lbp_bin_edges']

    _compressed = np.load("contrast_histogram.npz")
    contrast_histogram = _compressed['contrast_histogram']
    contrast_bin_edges = _compressed['contrast_bin_edges']

    idx = [random.randint(0, min(len(D), CUTOFF)) for _ in range(num_imgs)]

    ## LBP histograms
    fig, ax = plt.subplots(num_imgs, 3, constrained_layout=True)
    #  fig.suptitle("Original RGB Image (left), LBP image (middle), and histogram of intensities of the LBP image (right)")
    lbp_matrix = np.load(os.path.basename(DS_PATH) + f"_LBP.npy")
    for ax_cnt, i in enumerate(idx):
        hist = lbp_histograms[i, :]
        bin_edges = lbp_bin_edges[i, :]
        main_img = skimage.io.imread(D[i])
        lbp_img = lbp_matrix[i, :].reshape(config.SHAPE)
        ax[ax_cnt][0].imshow(main_img, aspect="auto")
        # Hide ticks (numbering) in axis
        ax[ax_cnt][0].set_xticklabels([])
        ax[ax_cnt][0].set_yticklabels([])
        ax[ax_cnt][0].set_xlabel("Original image (RGB)")

        ax[ax_cnt][1].imshow(lbp_img, cmap='gray', aspect="auto")
        # Hide ticks (numbering) in axis
        ax[ax_cnt][1].set_xticklabels([])
        ax[ax_cnt][1].set_yticklabels([])
        ax[ax_cnt][1].set_xlabel("LBP image (matrix of local binary patterns, interpreted as image)")

        ax[ax_cnt][2].stairs(hist, bin_edges, fill=True)
        # Hide ticks (numbering) in axis
        ax[ax_cnt][2].set_xlabel("Histogram of LBP image")

    plt.savefig("LBP_histograms.jpeg")
    print(f"Plotted LBP histograms")

    ## Contrast histogram
    fig, ax = plt.subplots(1, 1, constrained_layout=True)
    #  fig.suptitle("Original RGB Image (left), LBP image (middle), and histogram of intensities of the LBP image (right)")

    ax.stairs(contrast_histogram, contrast_bin_edges, fill=True)
    ax.set_xlabel("Histogram of distribution of contrasts across the first min(10000, size of all datasets combined) images")

    plt.savefig("contrast_histogram.jpeg")
    print("Plotted contrast histogram")
    print("Done")
