import skimage.exposure
import numpy as np
import argparse
from tqdm import tqdm
import config

def gen_lbp_histograms(lbp_files: list[str],
                       hist_file: str,
                       bins = 256,
                       normalize = True):
    hist_matrices = []
    bin_edges_matrices = []
    for i, lbp_filepath in enumerate(lbp_files):
        print(f"Working on lbp file {lbp_filepath}")
        lbp_matrix = np.load(lbp_filepath)
        h = np.zeros((lbp_matrix.shape[0], bins), dtype=np.float64)
        # b bins ==> b + 1 edges
        bin_edges = np.zeros((lbp_matrix.shape[0], bins + 1), dtype=np.float64)
        for j in tqdm(range(lbp_matrix.shape[0])):
            h[j, :], bin_edges_j = np.histogram(lbp_matrix[j, :], bins=bins)
            if normalize:
                bin_edges[j, :] = bin_edges_j.astype(np.float64) / np.float64(bin_edges_j.max())
            else:
                bin_edges[j, :] = bin_edges_j.astype(np.float64) 
        hist_matrices.append(h)
        bin_edges_matrices.append(bin_edges)

    print("Saving final histograms and respective bin_centers")
    final_hist_mat = np.concatenate(hist_matrices)
    final_bin_edges_matrix = np.concatenate(bin_edges_matrices)
    print(f"Final histogram matrix: shape {final_hist_mat.shape}")
    print(f"Final bin centers matrix: shape {final_bin_edges_matrix.shape}")
    np.savez_compressed(hist_file, lbp_histograms=final_hist_mat, lbp_bin_edges=final_bin_edges_matrix)
    print("Done")

def gen_global_contrast_histogram(contrast_files: list[str], 
                                  hist_file: str, 
                                  max_cnt: int = float('inf'),
                                  bins = 10,
                                  hist_range = None,
                                  idx_dist: int = 0,
                                  idx_angle: int = 0):
    # since constrast arrays are small, it is okay to load all at once
    c_mat= np.concatenate([np.load(filepath) for filepath in contrast_files], axis=0).reshape(-1, len(config.DISTANCES), len(config.ANGLES))
    print("Loading contrast files")
    truncated_c_arr = c_mat[0:min(c_mat.shape[0], max_cnt), idx_dist, idx_angle]
    print(f"Generated contrast array of shape {truncated_c_arr.shape}")

    final_hist, final_bin_edges = np.histogram(truncated_c_arr, bins=bins, range=hist_range)
    print(f"Generated histogram of shape {final_hist.shape}")
    print(f"\t and bin centers array of shape {final_bin_edges.shape}")
    print(f"Saving global contrast frequency histogram with {bins} bins (over whole dataset or over {max_cnt} items - whatever comes first)")
    np.savez_compressed(hist_file, contrast_histogram=final_hist, contrast_bin_edges=final_bin_edges)
    print("Done")

def main(lbp_files, constrast_files):
    gen_lbp_histograms(lbp_files, "LBP_histograms.npz")
    gen_global_contrast_histogram(constrast_files, "contrast_histogram.npz", max_cnt=10000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l','--lbp', nargs='+', help='List of LBP .npy files', required=True)
    parser.add_argument('-c','--contrast', nargs='+', help='List of constrast .npy files', required=True)
    
    args = parser.parse_args()
    main(args.lbp, args.contrast)
