from PIL import Image
import numpy as np

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


# Usage
input_image = "data\input_images\Screenshot 2024-09-18 162247.png"
output_image = "data\output_blue_images\Blue.png"

# Adjust the values below as needed
adjust_image(input_image, output_image, 
             red=0.1,        # Reduce red channel to 50%
             green=0.6,      # Reduce green channel to 70%
             blue=1.0,       # Keep blue channel at 100%
             brightness=0.5  # Increase overall brightness by 20%
            )