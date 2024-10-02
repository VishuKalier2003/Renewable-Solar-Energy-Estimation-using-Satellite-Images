import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# Function to open an image file
def open_file():
    global img, processed_img, hsv_img, img_display_size
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img_display_size = fit_image_to_window(img)  # Resize image to fit the display
        original_img = cv2.cvtColor(img_display_size, cv2.COLOR_BGR2RGB)  # Convert to RGB for display
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # Convert to HSV for green removal
        show_image(original_img, img_label_original)  # Show the original image in the left panel

# Function to remove green components from the image based on slider values
def remove_green():
    global processed_img

    # Get the slider values for HSV bounds
    lower_hue = lower_hue_slider.get()
    lower_sat = lower_sat_slider.get()
    lower_val = lower_val_slider.get()
    upper_hue = upper_hue_slider.get()
    upper_sat = upper_sat_slider.get()
    upper_val = upper_val_slider.get()

    # Define the range for green in HSV space based on the slider values
    lower_green = np.array([lower_hue, lower_sat, lower_val])  # Lower bound of green in HSV
    upper_green = np.array([upper_hue, upper_sat, upper_val])  # Upper bound of green in HSV

    # Create a mask to filter out the green
    mask = cv2.inRange(hsv_img, lower_green, upper_green)
    mask_inv = cv2.bitwise_not(mask)

    # Keep only the non-green parts of the image
    non_green_parts = cv2.bitwise_and(img, img, mask=mask_inv)
    processed_img = cv2.cvtColor(fit_image_to_window(non_green_parts), cv2.COLOR_BGR2RGB)  # Convert to RGB for display
    show_image(processed_img, img_label_processed)  # Show the processed image in the right panel

# Function to display the image on the provided label
def show_image(image, img_label):
    img = Image.fromarray(image)
    img_tk = ImageTk.PhotoImage(image=img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Function to resize image to fit the display window of 1980x1080 px
def fit_image_to_window(image):
    window_width = 960  # Each image width for half of 1920
    window_height = 540  # Each image height to fit within 1080
    h, w = image.shape[:2]
    
    # Resize keeping aspect ratio
    aspect_ratio = w / h
    if aspect_ratio > 1:  # Width > height
        new_w = window_width
        new_h = int(window_width / aspect_ratio)
    else:  # Height > width
        new_h = window_height
        new_w = int(window_height * aspect_ratio)

    return cv2.resize(image, (new_w, new_h))

# Function to save the processed image
def save_image():
    if processed_img is None:
        messagebox.showwarning("No Image", "Please process an image before saving.")
        return
    save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")])
    if save_path:
        # Convert the processed image back to BGR format for saving
        save_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, save_img)
        messagebox.showinfo("Image Saved", f"Image saved successfully at {save_path}")

# Create the Tkinter window
root = Tk()
root.title("Remove Green Components with Adjustable Threshold")
root.geometry("1980x1080")  # Set window size

# Create frames for the original and processed images
frame_left = Frame(root, width=960, height=540)
frame_left.pack(side=LEFT, padx=10, pady=10)

frame_right = Frame(root, width=960, height=540)
frame_right.pack(side=RIGHT, padx=10, pady=10)

# Create labels to display the images
img_label_original = Label(frame_left, text="Original Image")
img_label_original.pack()

img_label_processed = Label(frame_right, text="Processed Image (No Green)")
img_label_processed.pack()

# Create sliders for adjusting the lower and upper bounds of HSV values
lower_hue_slider = Scale(root, from_=0, to=180, orient=HORIZONTAL, label="Lower Hue", length=300)
lower_hue_slider.set(35)  # Default value for lower green hue
lower_hue_slider.pack(pady=5)

lower_sat_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Lower Saturation", length=300)
lower_sat_slider.set(40)  # Default value for lower green saturation
lower_sat_slider.pack(pady=5)

lower_val_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Lower Value", length=300)
lower_val_slider.set(40)  # Default value for lower green value
lower_val_slider.pack(pady=5)

upper_hue_slider = Scale(root, from_=0, to=180, orient=HORIZONTAL, label="Upper Hue", length=300)
upper_hue_slider.set(85)  # Default value for upper green hue
upper_hue_slider.pack(pady=5)

upper_sat_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Upper Saturation", length=300)
upper_sat_slider.set(255)  # Default value for upper green saturation
upper_sat_slider.pack(pady=5)

upper_val_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Upper Value", length=300)
upper_val_slider.set(255)  # Default value for upper green value
upper_val_slider.pack(pady=5)

# Create buttons for image processing
open_button = Button(root, text="Open Image", command=open_file)
open_button.pack(pady=10)

process_button = Button(root, text="Remove Green Components", command=remove_green)
process_button.pack(pady=10)

save_button = Button(root, text="Save Processed Image", command=save_image)
save_button.pack(pady=10)

# Initialize image variables
img = None
processed_img = None
hsv_img = None
img_display_size = None

# Start the Tkinter main loop
root.mainloop()
