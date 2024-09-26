from PIL import Image
import numpy as np
import os

def adjust_image(input_path, output_path, red=1.0, green=1.0, blue=1.0, brightness=1.0):
    """
    Adjust the intensity of each color channel in an image.
    
    :param input_path: Path to the input image
    :param output_path: Path to save the output image
    :param red: Intensity of red channel (0 to 1)
    :param green: Intensity of green channel (0 to 1)
    :param blue: Intensity of blue channel (0 to 1)
    :param brightness: Overall brightness adjustment (0 to 2, where 1 is original brightness)
    """
    # Open the image
    img = Image.open(input_path)
    
    # Convert image to RGB if it's not already
    img = img.convert('RGB')
    
    # Convert image to numpy array and normalize to float (0 to 1)
    img_array = np.array(img).astype(float) / 255

    # Adjust each channel
    img_array[:,:,0] *= red   # Red channel
    img_array[:,:,1] *= green # Green channel
    img_array[:,:,2] *= blue  # Blue channel

    # Adjust brightness
    img_array *= brightness

    # Clip values to ensure they're in the 0 to 1 range
    img_array = np.clip(img_array, 0, 1)

    # Convert back to 0-255 range and uint8 type
    img_array = (img_array * 255).astype(np.uint8)
    
    # Create new image from array
    output_img = Image.fromarray(img_array)
    
    # Save the output image
    output_img.save(output_path)


# Define the input and output folders
input_folder = "data/input_images"
output_folder = "data/output_blue_images"

# Define the color adjustments
red = 0.1
green = 0.6
blue = 1.0
brightness = 0.5

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    # Check if the file is an image
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Read the image
        input_path = os.path.join(input_folder, filename)
        output_filename = filename.split('.')[0] + "_blue.png"
        output_path = os.path.join(output_folder, output_filename)

        # Apply the color adjustments
        adjust_image(input_path, output_path, red, green, blue, brightness)

        print(f"Processed {filename} and saved as {output_filename}")