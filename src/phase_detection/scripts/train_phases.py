#!/usr/bin/env python3

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from phase_detection.dataset import PhaseDataset
from phase_detection.unet_model import UNet


def main():
    # project directories
    pkg = os.path.dirname(os.path.dirname(__file__))
    img_dir = os.path.join(pkg, "data", "images")
    mask_dir = os.path.join(pkg, "data", "masks")
    model_dir = os.path.join(pkg, "models")
    os.makedirs(model_dir, exist_ok=True)

    # dataset:
    #   train=True  → exclude test.png
    #   augment=True → flips, rotations, brightness jitter
    dataset = PhaseDataset(
        img_dir,
        mask_dir,
        img_size=512,
        train=True,
        augment=True
    )

    # single-sample stochastic training
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

    # CPU device (ARM container)
    device = torch.device("cpu")
    model = UNet(3, 3).to(device)  # 3 RGB channels → 3 classes (A,B,C)

    # CrossEntropy with weights: A(bg-lite)=1.0, B=2.0, C=2.0
    # This encourages model to pay attention to darker/mixed phases
    weights = torch.tensor([1.0, 2.0, 2.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # fewer epochs, faster training, more robust because of augmentation
    num_epochs = 60
    model_path = os.path.join(model_dir, "unet_phases.pth")

    for epoch in range(num_epochs):
        model.train()
        tr_loss = 0.0

        for img, mask in train_loader:
            img, mask = img.to(device), mask.to(device)

            out = model(img)
            loss = criterion(out, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_loss += loss.item()

        print(f"EPOCH {epoch} TRAIN LOSS {tr_loss:.4f}")

    # save last model (stable training, no noisy val)
    torch.save(model.state_dict(), model_path)
    print("Training complete. Saved model at:", model_path)


if __name__ == "__main__":
    main()
