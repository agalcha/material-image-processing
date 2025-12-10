import json
import numpy as np
import cv2
import glob
import os

# label names EXACTLY from label_names.txt
label_to_id = {
    "_background_": 0,
    "A": 1,
    "B": 2,
    "C": 3,
}

# find all jsons in this folder
json_files = glob.glob("*.json")

# create masks folder one level up
output_dir = os.path.join("..", "masks")
os.makedirs(output_dir, exist_ok=True)

for jf in json_files:
    print("Processing:", jf)
    with open(jf, "r") as f:
        data = json.load(f)

    h = data["imageHeight"]
    w = data["imageWidth"]

    # blank mask
    mask = np.zeros((h, w), dtype=np.uint8)

    # draw each polygon
    for shape in data["shapes"]:
        label = shape["label"]
        pts = np.array(shape["points"], dtype=np.int32)
        class_id = label_to_id[label]
        cv2.fillPoly(mask, [pts], class_id)

    # save filename like img1.png
    name = jf.replace(".json", ".png")
    out_path = os.path.join(output_dir, name)
    cv2.imwrite(out_path, mask)

    print("  unique:", np.unique(mask))
