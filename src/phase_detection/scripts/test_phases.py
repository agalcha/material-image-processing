#!/usr/bin/env python3

import os
import torch
import numpy as np
import cv2

from phase_detection.unet_model import UNet


def main():
    pkg = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(pkg, "models", "unet_phases.pth")
    
    test_image = os.path.join(pkg, "data", "images", "test.png")

    img_orig = cv2.imread(test_image)
    h, w = img_orig.shape[:2]

    img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (512, 512))
    arr = img.astype(np.float32)/255.0
    arr = np.transpose(arr, (2,0,1))
    t = torch.from_numpy(arr).unsqueeze(0)

    model = UNet(3,3)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    with torch.no_grad():
        out = model(t)
        mask_small = torch.argmax(out, dim=1).squeeze().numpy()

    mask = cv2.resize(mask_small.astype(np.uint8), (w,h), interpolation=cv2.INTER_NEAREST)

    total = mask.size
    pctA = (mask==0).sum()/total
    pctB = (mask==1).sum()/total
    pctC = (mask==2).sum()/total

    print("A:", pctA*100, "%")
    print("B:", pctB*100, "%")
    print("C:", pctC*100, "%")

    colors = [(0,255,255),(255,0,0),(255,0,255)]

    #colored segmentation overlay
    overlay = img_orig.copy()
    for i,c in enumerate(colors):
        overlay[mask==i]=c

    blend = cv2.addWeighted(img_orig, 0.6, overlay, 0.4, 0)
    cv2.imwrite(os.path.join(pkg,"models","test_overlay.png"), blend)


if __name__ == "__main__":
    main()
