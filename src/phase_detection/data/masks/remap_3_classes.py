import cv2
import numpy as np
import glob

for f in glob.glob("*.png"):
    mask = cv2.imread(f, 0)

    # merge background with A
    mask[mask == 0] = 1

    # shift values: 1→0, 2→1, 3→2
    mask = mask - 1

    cv2.imwrite(f, mask)
    print(f, np.unique(mask))
