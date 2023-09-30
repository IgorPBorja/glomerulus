import numpy as np

IGNORE_DIRS = ["Crescente"]
SHAPE = (768, 1024)
P = 8
R = 1.0
DISTANCES = [1]
ANGLES = [0.0, np.pi * 0.25, np.pi * 0.5, np.pi * 0.75, np.pi]
MAXL = 2**8 - 1
CUTOFF = float('inf') ## NOTE in final execution it should always be infinity (float('inf'))
