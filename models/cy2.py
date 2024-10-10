# %%
import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
import os
import glob
import cv2
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
import numpy as np
from tensorflow.keras.callbacks import TensorBoard # type: ignore
import datetime
import seaborn
import matplotlib.pyplot as plt
import pandas as pd

# %%
# YOLO Block: Custom convolutional block with Conv + BatchNorm + LeakyReLU
def yolo_block(inputs, filters):
    x = layers.Conv2D(filters, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

# %%
# YOLO Model Definition
def YOLO(input_shape, num_classes, num_anchors):
    inputs = layers.Input(shape=input_shape)

    # YOLO architecture layers
    x = yolo_block(inputs, 32)
    x = layers.MaxPooling2D((2, 2))(x)

    x = yolo_block(x, 64)
    x = layers.MaxPooling2D((2, 2))(x)

    x = yolo_block(x, 128)
    x = yolo_block(x, 64)
    x = yolo_block(x, 128)
    x = layers.MaxPooling2D((2, 2))(x)

    x = yolo_block(x, 256)
    x = yolo_block(x, 128)
    x = yolo_block(x, 256)
    x = layers.MaxPooling2D((2, 2))(x)

    x = yolo_block(x, 512)
    x = yolo_block(x, 256)
    x = yolo_block(x, 512)
    x = layers.MaxPooling2D((2, 2))(x)

    x = yolo_block(x, 1024)
    x = yolo_block(x, 512)
    x = yolo_block(x, 1024)

    # YOLO Head: Final Conv layer to predict the bounding boxes and class probabilities
    x = layers.Conv2D(num_anchors * (num_classes + 5), (1, 1), padding='same')(x)

    # Reshape the output to (grid_size, grid_size, num_anchors, 5 + num_classes)
    outputs = layers.Reshape((input_shape[0] // 32, input_shape[1] // 32, num_anchors, num_classes + 5))(x)

    # Define the model
    model = Model(inputs, outputs)

    return model

# %%
def yolo_loss(y_true, y_pred):
    # y_true and y_pred are expected to have the shape (batch_size, grid_size, grid_size, num_anchors, 5 + num_classes)
    
    # Extract the predicted bounding boxes and class probabilities
    pred_box = y_pred[..., 0:4]
    pred_confidence = y_pred[..., 4:5]
    pred_class_probs = y_pred[..., 5:]

    # Extract the ground truth bounding boxes and class probabilities
    true_box = y_true[..., 0:4]
    true_confidence = y_true[..., 4:5]
    true_class_probs = y_true[..., 5:]

    # Loss for bounding box coordinates (localization loss)
    coord_loss = tf.reduce_mean(tf.square(true_box - pred_box))
    
    # Loss for confidence scores (objectness loss)
    conf_loss = tf.reduce_mean(tf.square(true_confidence - pred_confidence))
    
    # Loss for class probabilities (classification loss)
    class_loss = tf.reduce_mean(tf.square(true_class_probs - pred_class_probs))

    # Combine the losses
    total_loss = coord_loss + conf_loss + class_loss
    
    return total_loss

# %%
# Define model parameters
input_shape = (640, 640, 3)  # YOLO typically uses 640x640 images
num_classes = 1  # Number of object classes (solar-panel)
num_anchors = 5  # Number of anchor boxes

# %%
# Build the YOLO model
yolo_model = YOLO(input_shape, num_classes, num_anchors)

# %%
# Compile the model with the custom loss function
yolo_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                   loss=yolo_loss)

# %%
# Model summary
yolo_model.summary()

# %%
def load_yolo_dataset(image_dir, label_dir):
    image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('.png')])
    label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir) if lbl.endswith('.txt')])

    images = []
    labels = []

    for img_path, lbl_path in zip(image_paths, label_paths):
        # Load the image
        image = load_img(img_path, target_size=(640, 640))  # Resize to match YOLO input shape
        image = img_to_array(image) / 255.0  # Normalize image data
        images.append(image)

        # Load the label
        with open(lbl_path, 'r') as f:
            label = f.readlines()
            label = np.array([list(map(float, line.strip().split())) for line in label])
            labels.append(label)

    return np.array(images), np.array(labels)


# %%
# Set up TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# %%
# Custom Callback to Collect Metrics
class TrainingHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.epoch_losses = []
        self.epoch_val_losses = []

    def on_epoch_end(self, epoch, logs=None):
        # Collect losses at the end of each epoch
        self.epoch_losses.append(logs.get('loss'))
        self.epoch_val_losses.append(logs.get('val_loss'))

# Instantiate the callback
training_history = TrainingHistory()

# %%
import os
import yaml

# Load the dataset configuration from data.yaml
with open("./data/Solar-panel-detection.v3i.yolov8-obb/data.yaml") as file:
    config = yaml.safe_load(file)

# Get paths from the loaded configuration
train_images_dir = "./data/Solar-panel-detection.v3i.yolov8-obb/train/images"
train_labels_dir = "./data/Solar-panel-detection.v3i.yolov8-obb/train/labels"
valid_images_dir = "./data/Solar-panel-detection.v3i.yolov8-obb/valid/images"
valid_labels_dir = "./data/Solar-panel-detection.v3i.yolov8-obb/valid/labels"

# Load the train and validation datasets
train_images, train_labels = load_yolo_dataset(train_images_dir, train_labels_dir)
valid_images, valid_labels = load_yolo_dataset(valid_images_dir, valid_labels_dir)

# Train the YOLO model
yolo_model.fit(train_images, train_labels, validation_data=(valid_images, valid_labels),
               batch_size=16, epochs=50, callbacks=[tensorboard_callback, training_history])


# %% [markdown]
# **Launching TensorBoard**
# 
# After starting the training process, you can use TensorBoard to visualize training metrics in real-time.
# 
# ```bash
# tensorboard --logdir logs/fit
# ```
# Open the URL provided by TensorBoard in your browser to view metrics like loss, accuracy, and others.
# 

# %%
# Evaluate on test set
test_images, test_labels = load_yolo_dataset('data/Solar-panel-detection.v3i.yolov8-obb/test')
loss = yolo_model.evaluate(test_images, test_labels)
print(f"Test loss: {loss}")

# %%
# Prepare the data for plotting
history_df = pd.DataFrame({
    'Epoch': range(1, len(training_history.epoch_losses) + 1),
    'Training Loss': training_history.epoch_losses,
    'Validation Loss': training_history.epoch_val_losses
})

# Plot using Seaborn
plt.figure(figsize=(10, 6))
sns.lineplot(x='Epoch', y='Training Loss', data=history_df, label='Training Loss', marker='o')
sns.lineplot(x='Epoch', y='Validation Loss', data=history_df, label='Validation Loss', marker='o')

# Customize the plot
plt.title('Training vs Validation Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(loc='best')
plt.grid(True)
plt.show()



