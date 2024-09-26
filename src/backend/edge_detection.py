import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the input and output folders
input_folder = "data\output_blue_images"
output_folder = "data\output_hed_images"

# Define the model paths
protopath = "weights/hed/deploy.prototxt"
modelpath = "weights/hed/hed_pretrained_bsds.caffemodel"

# Load the model
net = cv2.dnn.readNetFromCaffe(protopath, modelpath)

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Read the image
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Apply the same processing as before
        sigma = 0.3
        median = np.median(image)
        (H, W) = image.shape[:2]
        mean_pixel_values = np.average(image, axis=(0, 1))
        blob = cv2.dnn.blobFromImage(image, scalefactor=2, size=(W, H), mean=(mean_pixel_values[0]+40, mean_pixel_values[1]+40, mean_pixel_values[2]+40), swapRB=True, crop=False)

        # Get the output from the model
        net.setInput(blob)
        hed = net.forward()
        hed = cv2.resize(hed[0,0], (W, H))
        hed = (255 * hed).astype("uint8")

        # Save the output image
        output_filename = filename.split('.')[0] + "_hed.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, hed)

        print(f"Processed {filename} and saved as {output_filename}")