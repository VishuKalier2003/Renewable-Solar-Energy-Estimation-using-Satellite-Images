import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

# YOLO Block: Custom convolutional block with Conv + BatchNorm + LeakyReLU
def yolo_block(inputs, filters):
    x = layers.Conv2D(filters, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    return x

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


# Define model parameters
input_shape = (640, 640, 3)  # YOLO typically uses 416x416 images
num_classes = 6  # Number of object classes
num_anchors = 5  # Number of anchor boxes

# Build the YOLO model
yolo_model = YOLO(input_shape, num_classes, num_anchors)

# Compile the model with the custom loss function
yolo_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                   loss=yolo_loss)

# Model summary
yolo_model.summary()


'''
place the sinppet below in the train.py to access the model
yolo_model.fit(train_data, train_labels, epochs=50, batch_size=32, validation_data=(val_data, val_labels))
'''
