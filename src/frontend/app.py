import tkinter as tk
from tkinter import filedialog
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading

class ImageProcessorApp(ttk.Window):
    def __init__(self):
        super().__init__(themename="darkly")
        self.title("Advanced Image Processor")
        self.geometry("1920x1080")

        self.image_path = None
        self.weights_path = None
        self.processed_image = None

        self.create_widgets()

    def create_widgets(self):
        # Main fram
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=BOTH, expand=YES, padx=10, pady=10)

        # Image selection frame
        image_frame = ttk.LabelFrame(main_frame, text="Image Selection", padding=10)
        image_frame.pack(fill=X, pady=10)

        self.select_image_button = ttk.Button(image_frame, text="Select Image", command=self.select_image, style='Accent.TButton')
        self.select_image_button.pack(side=LEFT, padx=5)

        self.image_path_label = ttk.Label(image_frame, text="No image selected")
        self.image_path_label.pack(side=LEFT, padx=5, fill=X, expand=YES)

        # Image display frame
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=BOTH, expand=YES, pady=10)

        self.original_image_label = ttk.Label(display_frame, text="Original Image")
        self.original_image_label.pack(side=LEFT, padx=10, expand=YES)

        self.processed_image_label = ttk.Label(display_frame, text="Processed Image")
        self.processed_image_label.pack(side=RIGHT, padx=10, expand=YES)

        # Weights selection frame
        weights_frame = ttk.LabelFrame(main_frame, text="Weights Selection", padding=10)
        weights_frame.pack(fill=X, pady=10)
        
        '''
        WEIGHTS NEED TO BE LINKED WITH THE FRONTEND PROGRAM
        '''
        weights_options = [
            ("HED Weights", "weights/hed/hed_pretrained_bsds.caffemodel"),
            ("Other Weights 1", "path/to/other_weights1.caffemodel"),
            ("Other Weights 2", "path/to/other_weights2.caffemodel")
        ]

        for text, weights in weights_options:
            button = ttk.Button(weights_frame, text=text, 
                                command=lambda w=weights: self.select_weights(w),
                                style='Accent.TButton')
            button.pack(side=LEFT, padx=5)

        # Process button
        self.process_button = ttk.Button(main_frame, text="Process Image", 
                                         command=self.process_image, 
                                         style='success.Accent.TButton')
        self.process_button.pack(pady=10)

        # Status bar
        self.status_var = tk.StringVar()
        self.status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=W)
        self.status_bar.pack(side=BOTTOM, fill=X)

    def select_image(self):
        self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if self.image_path:
            self.image_path_label.config(text=self.image_path)
            self.display_image(self.image_path, self.original_image_label)

    def display_image(self, image_path, label):
        image = Image.open(image_path)
        image.thumbnail((900, 900))  # Resize for display
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.image = photo

    def select_weights(self, weights_path):
        self.weights_path = weights_path
        self.status_var.set(f"Selected weights: {weights_path}")

    def process_image(self):
        if not self.image_path or not self.weights_path:
            self.status_var.set("Please select both an image and weights.")
            return

        # Here you would typically do the image processing
        # For this example, we'll just display the original image in both places
        self.display_image(self.image_path, self.processed_image_label)
        self.status_var.set("Image processed successfully!")

# This block should be at the same indentation level as the class definition
if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()
