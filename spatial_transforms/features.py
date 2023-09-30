import sys, os
import skimage
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")  # supresses output (preventing it from opening)
# since the Agg backend can't show pictures
# see https://stackoverflow.com/questions/65101357/matplotlib-prevent-plot-from-showing

def gen_channels_histograms(img: np.ndarray, nbins: int = 256, path: str = None, num_channels: int = 3, channel_names=["R", "G", "B"]) -> None:
    """
        Generate histograms for all channels (plotted side by side)
    """
    fig, axis = plt.subplots(1, num_channels)
    fig.suptitle(f"{''.join(channel_names)} histogram, separated channel-wise")
    fig.tight_layout()  # better spacing between subplots
    for i, channel in zip(range(num_channels), channel_names):
        hist_array, bin_centers = skimage.exposure.histogram(img[:, :, i], nbins=nbins, normalize=True)
        # normalize = True ==> normalize frequencies to [0, 1], by divinding
        hist_array = np.pad(hist_array, (bin_centers[0], 255 - bin_centers[-1]), mode="constant")
        axis[i].set_xticks([64*i for i in range(5)])  # divide labeling x axis in 4
        axis[i].bar(range(256), hist_array)
        axis[i].set_title(f"Channel {channel} of {''.join(channel_names)}")
        if i == num_channels // 2:  # set xlabel only on middle x-axis
            axis[i].set_xlabel("Bins (arranged by their centers)")
        if i == 0:  # set ylabel only on first y-axis
            axis[i].set_ylabel("Frequencies (normalized)")
    if path is not None:
        plt.savefig(path)
    plt.show()

def gen_opponent(img: np.ndarray, path: str) -> None:
    """
        In RGB image:
        O1 = (R - G) / sqrt(2)
        O2 = (R + G - 2B)/ sqrt(6)
        O3 = (R + G + B)/sqrt(3)
        Assumes intensity values in range [0, 1]
    """
    float_img = skimage.util.img_as_float(img)
    O1 = (float_img[:, :, 0] - float_img[:, :, 1]) / np.sqrt(2)
    O2 = (float_img[:, :, 0] + float_img[:, :, 1] - 2.0 * float_img[:, :, 2]) / np.sqrt(6)
    O3 = (float_img[:, :, 0] + float_img[:, :, 1] + float_img[:, :, 2]) / np.sqrt(3)
    non_scaled_opponent = np.concatenate([O1.reshape(*O1.shape, 1), O2.reshape(*O2.shape, 1), O3.reshape(*O3.shape, 1)], axis=-1)

    final_opponent = skimage.util.img_as_ubyte(np.clip(non_scaled_opponent, 0.0, 1.0))
    skimage.io.imsave(path, final_opponent)

def color_transformed(img: np.ndarray, num_channels: int = 3) -> np.ndarray:
    """
        Normalize color channels to mean 0 and variance 1. Assumes intensity values in range [0, 1]
    """
    float_img = skimage.util.img_as_float(img)
    for i in range(num_channels):
        mu_i = np.mean(float_img[:, :, i])
        sigma_i = np.std(float_img[:, :, i])
        float_img[:, :, i] = np.clip((float_img[:, :, i] - mu_i) / sigma_i, 0.0, 1.0)
    return float_img



