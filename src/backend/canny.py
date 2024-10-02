import cv2
import numpy as np
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk

# Set max window size (1920x1080, keeping some space for widgets)
MAX_WINDOW_WIDTH = 1920
MAX_WINDOW_HEIGHT = 1080
IMAGE_FRAME_WIDTH = MAX_WINDOW_WIDTH // 2
IMAGE_FRAME_HEIGHT = MAX_WINDOW_HEIGHT - 100  # Leaving some space for sliders and buttons

# Function to open an image file
def open_file():
    global img, original_img, edges_img
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image
        img = cv2.imread(file_path)
        original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Tkinter display
        original_img_resized = resize_image(original_img, IMAGE_FRAME_WIDTH, IMAGE_FRAME_HEIGHT)  # Resize image
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for Canny edge detection
        edges_img = gray_img.copy()
        show_image(original_img_resized, img_label_original)  # Show original image
        apply_canny(100)  # Apply initial Canny edge detection with default threshold

# Function to resize image to fit within the window while maintaining aspect ratio
def resize_image(image, max_width, max_height):
    h, w, _ = image.shape
    aspect_ratio = w / h
    if w > max_width:
        w = max_width
        h = int(w / aspect_ratio)
    if h > max_height:
        h = max_height
        w = int(h * aspect_ratio)
    return cv2.resize(image, (w, h))

# Function to display the image on the provided label
def show_image(image, img_label):
    img = Image.fromarray(image)
    img_tk = ImageTk.PhotoImage(image=img)
    img_label.config(image=img_tk)
    img_label.image = img_tk

# Function to apply Canny edge detection and update the edge-detected image
def apply_canny(threshold1):
    global edges_img
    edges = cv2.Canny(edges_img, threshold1, threshold1 * 2)  # Apply Canny edge detection
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)  # Convert edges to RGB for display
    edges_resized = resize_image(edges_rgb, IMAGE_FRAME_WIDTH, IMAGE_FRAME_HEIGHT)  # Resize for display
    show_image(edges_resized, img_label_edges)  # Show the edges

# Create the Tkinter window
root = Tk()
root.title("Canny Edge Detection with Real-time Slider")
root.geometry(f"{MAX_WINDOW_WIDTH}x{MAX_WINDOW_HEIGHT}")

# Create frames for the original and edge-detected images
frame_left = Frame(root, width=IMAGE_FRAME_WIDTH, height=IMAGE_FRAME_HEIGHT)
frame_left.pack(side=LEFT, padx=10, pady=10)
frame_right = Frame(root, width=IMAGE_FRAME_WIDTH, height=IMAGE_FRAME_HEIGHT)
frame_right.pack(side=RIGHT, padx=10, pady=10)

# Create labels to display the images
img_label_original = Label(frame_left, text="Original Image")
img_label_original.pack()

img_label_edges = Label(frame_right, text="Edge-detected Image")
img_label_edges.pack()

# Create a slider to adjust the Canny threshold
threshold_slider = Scale(root, from_=0, to=255, orient=HORIZONTAL, label="Canny Threshold 1",
                         command=lambda val: apply_canny(int(val)))
threshold_slider.set(100)  # Set default threshold
threshold_slider.pack()

# Create a button to open an image file
open_button = Button(root, text="Open Image", command=open_file)
open_button.pack()

# Initialize image variables
img = None
original_img = None
edges_img = None

# Start the Tkinter main loop
root.mainloop()
