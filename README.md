# Phase Detection with UNet🤖

This is a project that uses UNet, a type of convolutional neural network (CNN) architecture that is used primarily for image segmentation tasks in machine learning to detect different phases in a microscopic image of a material.  
UNet works great with small datasets! I used LabelMe to take polygon shaped samples of the different patches in each image and used the pixels inside each polygon to train my model to identify the differences.  
The final outpuy is an overlay of the test image with all the phases highlighted in different colours.  
