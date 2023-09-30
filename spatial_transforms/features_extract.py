from sys import argv
import features
import numpy as np
import skimage

scalingT = 1.25
shiftT = 0.10

suffix_change = "_change"
suffix_shift = "_shift"
suffix_change_shift = "_change_shift"

def transform(
        img: np.ndarray,
        s: float,
        delta: float):
    """Applies light intensity change and shift. Assumes image as float (in [0,1] range). Then, processes image."""
    return np.clip(s * img + delta, 0.0, 1.0)

def save_transformed(img, srcpath: str, mode: str):
    if (mode == "change"):
        new_img = skimage.util.img_as_ubyte(transform(img, scalingT, 0.0))
    elif (mode == "shift"):
        new_img = skimage.util.img_as_ubyte(transform(img, 1.0, shiftT))
    elif (mode == "change+shift"):
        new_img = skimage.util.img_as_ubyte(transform(img, scalingT, shiftT))
    elif (mode == ""):
        new_img = skimage.util.img_as_ubyte(img)
    else:
        raise ValueError("Invalid mode")

    skimage.io.imsave(srcpath, new_img)

def process(
        srcpath: str, 
        histogram_destpath: str,
        normalized_destpath: str,
        opponent_destpath: str,
        opponent_hist_destpath: str):
    img = skimage.io.imread(srcpath)
    features.gen_channels_histograms(img, nbins=256, path=histogram_destpath)

    normalized_img = features.color_transformed(skimage.util.img_as_float(img))
    skimage.io.imsave(normalized_destpath, skimage.util.img_as_ubyte(normalized_img))

    features.gen_opponent(img, path=opponent_destpath)
    opponent_img = skimage.io.imread(opponent_destpath)
    features.gen_channels_histograms(opponent_img, nbins=256, path=opponent_hist_destpath)

def main():
    srcpath, histogram_destpath, normalized_destpath, \
            opponent_destpath, opponent_hist_destpath = argv[1:6]

    img = skimage.io.imread(srcpath)
    img = skimage.util.img_as_float(img)

    for p in argv[1:6]:
        assert(p.endswith(".jpg"))

    for suffix, mode in zip(["", suffix_change, suffix_shift, suffix_change_shift], ["", "change", "shift", "change+shift"]):
        save_transformed(img, srcpath.replace(".jpg", suffix+".jpg"), mode=mode)

        process(srcpath.replace(".jpg", suffix+".jpg"), histogram_destpath.replace(".jpg", suffix+".jpg"), normalized_destpath.replace(".jpg", suffix+".jpg"), opponent_destpath.replace(".jpg", suffix+".jpg"), opponent_hist_destpath.replace(".jpg", suffix+".jpg"))

if __name__ == "__main__":
    main()
