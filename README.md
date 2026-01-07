# Phase Detection Using U-Net in ROS🤖

Semantic segmentation pipeline to detect different phases in microscopic images using U-Net, deployed as a ROS node. Outputs a color overlay and percentage of each phase.

## Features  
Real-time inference on small datasets using patch-based training and augmentation.  
ROS integration with cv_bridge for image publishing and visualization.  
ARM64-native Docker setup optimized for Apple Silicon (fast PyTorch & OpenCV).  

## Quick Setup
Docker + ROS
<pre># Pull ARM64-native ROS 1 Noetic base image
docker pull ros:noetic-ros-base

# Run container
docker run -it --name ros1 -v ~/ip_project:/root/catkin_ws ros:noetic-ros-base bash

# Install desktop tools inside container
apt update && apt install ros-noetic-desktop-full -y </pre>

## Setup Catkin workspace:

<pre>mkdir -p /root/catkin_ws/src
cd /root/catkin_ws
catkin_make
echo "source /root/catkin_ws/devel/setup.bash" >> ~/.bashrc
source /root/catkin_ws/devel/setup.bash </pre>

## Python & ML Dependencies
<pre>apt update
apt install python3-pip python3-catkin-tools -y
pip3 install numpy pillow opencv-python
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
apt install ros-noetic-cv-bridge ros-noetic-image-transport -y</pre>

## Project Structure
phase_detection/
├─ data/        # images + masks  
├─ models/      # trained weights (unet_phases.pth)  
├─ scripts/     # train_phases.py, test_phases.py  
└─ src/phase_detection/  
   ├─ dataset.py  
   └─ unet_model.py  


dataset.py → loads images/masks as PyTorch tensors  
unet_model.py → defines U-Net architecture  
train_phases.py → trains on 3 images, saves best model  
test_phases.py → runs inference on test image, outputs overlay + phase percentages  

## Usage

Train the model
<pre>rosrun phase_detection train_phases.py</pre>

Test the model
<pre>rosrun phase_detection test_phases.py</pre>

Generates models/test_overlay.png

Computes percentage of each phase

## How It Works

Annotate images with LabelMe (polygons for each phase).  
Load image-mask pairs using PhaseDataset.  
Train U-Net on 3 images, validate on 1 unseen image.  
Save best weights (unet_phases.pth).  
Run inference → overlay + phase percentages.  

**Why U-Net?**  
Efficient with small datasets, captures local & global features via skip connections. Standard for microscopy & materials science.
