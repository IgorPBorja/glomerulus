import skimage
import skimage.feature, skimage.filters, skimage.color, skimage.transform, skimage.util
from dataset import Dataset
import numpy as np
from tqdm import tqdm
import sys, os.path
import config

"""
LBP format:
    Parameters:
        H x W images (a total of N)
        P = 8: number of neighbors
        R = 1.0: radius
    Output:
        matrix (N, H x W)
        where H and W are the standard sizes established
        (see variable shape)
        
        This is the result of flattening each H x W image.
    
GLCM format:
    Parameters:
        N images H x W
        distances: list of D magnitudes of offsets
        angles: list of A angles of offsets 
    Output:
        2-D array of shape (N, MAXL * MAXL * D * A)
        where MAXL is the maximum intensity (with 8-byte unsigned int images it is 2^8 - 1 = 255)
        
        this is the result of flattening the 5D array
        of shape (N, MAXL, MAXL, D, A) which is the natural output of applying GLCM N times

SOBEL format:
    Parameters: 
        N images H x W
    Output:
        2-D array of shape (N, H x W)
        
        this is the result of flattening each filtered H x W image
"""

if __name__ == '__main__':
    DS_PATH = sys.argv[1]
    if (len(sys.argv) > 2):
        CUTOFF = float(sys.argv[2])
    else:
        CUTOFF = float('inf')
    lbp_path = os.path.basename(DS_PATH) + "_LBP.npy" 
    glcm_path = os.path.basename(DS_PATH) + "_GLCM.npy"
    sobel_path = os.path.basename(DS_PATH) + "_sobel.npy"
    contrast_path = os.path.basename(DS_PATH) + "_contrast.npy"

    transforms = [
        ## standard LBP
        ## LBP generates float64 image in range [0.0, 255.0]
        lambda img: skimage.feature.local_binary_pattern(
            img,
            P = config.P,
            R = config.R,
            method="default"
        ).astype(np.uint8),

        ## GLCM matrix
        ## return type: np.uint32
        ## we do rescaling (mapping [0, (maxL + 1)**2] to [0, 2**32 - 1])
        ## then normalize to float interval [0, 1]
        ## horizontal and diagonal offset 
        lambda img: 
        skimage.util.img_as_float(
            skimage.exposure.rescale_intensity(
            skimage.feature.graycomatrix(
                image=img,
                distances=config.DISTANCES,
                angles=config.ANGLES
                )
            )
        ),
       
        ## Sobel filter
        ## return type: float in [0.0, 1.0]
        skimage.filters.sobel
    ]

    ds: Dataset = Dataset(DS_PATH, config.IGNORE_DIRS)

    LBP_features = np.zeros([min(len(ds), CUTOFF), np.prod(config.SHAPE)], dtype=np.uint8)

    ## GLCM produces np.uint32 output
    ## see https://github.com/scikit-image/scikit-image/blob/main/skimage/feature/texture.py
    ## MAXL + 1 = #{0, ..., MAXL}
    GLCM_features = np.zeros([min(len(ds), CUTOFF), (config.MAXL + 1) * (config.MAXL + 1) * len(config.DISTANCES) * len(config.ANGLES)], dtype=np.uint32)

    sobel_features = np.zeros([min(len(ds), CUTOFF), np.prod(config.SHAPE)], dtype=np.uint8)
    contrast_features = np.zeros([min(len(ds), CUTOFF), len(config.DISTANCES) * len(config.ANGLES)])

    for i, T in enumerate(transforms): 
        T_preprocessed = [
            lambda a : skimage.color.rgb2gray(a) if (len(a.shape) == 3 and a.shape[-1] == 3) else a, ## map to grayscale (if not in grayscale)
            lambda a: skimage.transform.resize(a, config.SHAPE), ## resize to SHAPE
            skimage.util.img_as_ubyte, ## rgb2gray uses float coeficients, so the resulting image is float. Also, skimage "resize" uses interpolation (producing float image)
            T, ## apply transform
            skimage.util.img_as_ubyte, ## sobel filter transforms image to float
            lambda a: np.reshape(a, [np.prod(a.shape)]), ## flatten (total)
        ]
        for j, res in enumerate(tqdm(ds.lazy_apply(T_preprocessed, as_float=False), total=min(len(ds), CUTOFF))):
            if (j >= CUTOFF): break
            if (i == 0):
                LBP_features[j, :] = res
            elif (i == 1):
                GLCM_features[j, :] = res
            elif (i == 2):
                sobel_features[j, :] = res

    for j in tqdm(range(min(len(ds), CUTOFF))):
        contrast_row_j = skimage.feature.graycoprops(GLCM_features[j, :].reshape(config.MAXL + 1, config.MAXL + 1, len(config.DISTANCES), len(config.ANGLES)), prop="contrast")
        contrast_features[j, :] = contrast_row_j.reshape(len(config.DISTANCES) * len(config.ANGLES))


    print("Extracted features")
    print(f"LBP: {LBP_features.shape}")
    print(f"GLCM: {GLCM_features.shape}")
    print(f"Sobel: {sobel_features.shape}")
    print(f"Contrast matrix: {contrast_features.shape}")

    print(f"Saving LBP in {lbp_path}")
    np.save(lbp_path, LBP_features)

    print(f"Saving GLCM in {glcm_path}")
    np.save(glcm_path, GLCM_features)

    print(f"Saving Sobel-filtered images in {sobel_path}")
    np.save(sobel_path, sobel_features)

    print(f"Saving contrast matrix in {contrast_path}")
    np.save(contrast_path, contrast_features)

    print("Done!")
