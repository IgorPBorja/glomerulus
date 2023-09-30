import sys, os
import skimage
import numpy as np
import matplotlib.pyplot as plt

def gen_channels_histograms(img: np.ndarray, nbins=256, path: str = None, num_channels: int = 3, channel_names=["R", "G", "B"]):
    """
        Generate histograms for all channels (plotted side by side)
    """
    fig, axis = plt.subplots(1, num_channels)
    fig.suptitle(f"{''.join(channel_names)} histogram, separated channel-wise")
    for i, channel in zip(range(num_channels), channel_names):
        hist_array, bin_centers = skimage.exposure.histogram(img[:,:, i], nbins=nbins, normalize=True)
        ## normalize = True ==> normalize frequencies to [0,1], by divinding
        axis[i].bar(bin_centers, hist_array)
        axis[i].set_title(f"Frequency histogram for channel {channel} of {''.join(channel_names)} color representation model")
        axis[i].set_xlabel("Bins (arranged by their centers)")
        axis[i].set_ylabel("Frequencies (normalized)")
    if path is not None:
        plt.savefig(path)
    plt.show()

def gen_O12(img: np.ndarray):
    pass

def color_transformed(img: np.ndarray):
    pass



