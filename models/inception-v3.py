# structure of a GoogleNet (Inception V3) model
import tensorflow as tf
from tensorflow.keras import layers, Model, Input

# Inception Module
def inception_module(x, f1, f3_r, f3, f5_r, f5, fpool):
    # 1x1 conv
    conv1 = layers.Conv2D(f1, (1,1), padding='same', activation='relu')(x)

    # 3x3 conv
    conv3_r = layers.Conv2D(f3_r, (1,1), padding='same', activation='relu')(x)
    conv3 = layers.Conv2D(f3, (3,3), padding='same', activation='relu')(conv3_r)

    # 5x5 conv
    conv5_r = layers.Conv2D(f5_r, (1,1), padding='same', activation='relu')(x)
    conv5 = layers.Conv2D(f5, (5,5), padding='same', activation='relu')(conv5_r)

    # Pooling
    pool = layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(x)
    pool_conv = layers.Conv2D(fpool, (1,1), padding='same', activation='relu')(pool)

    # Concatenate all filters
    output = layers.concatenate([conv1, conv3, conv5, pool_conv], axis=-1)
    
    return output

# GoogleNet (Inception V3) Architecture Definition
input_layer = Input(shape=(299, 299, 3))

# Initial Conv and MaxPool Layers
x = layers.Conv2D(32, (3,3), strides=(2,2), padding='valid', activation='relu')(input_layer)
x = layers.Conv2D(32, (3,3), padding='valid', activation='relu')(x)
x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D((3,3), strides=(2,2))(x)

# Inception modules
x = inception_module(x, 64, 48, 64, 8, 16, 32)
x = inception_module(x, 64, 48, 64, 8, 16, 32)
x = inception_module(x, 128, 64, 128, 16, 32, 64)
x = layers.MaxPooling2D((3,3), strides=(2,2))(x)

x = inception_module(x, 128, 64, 128, 16, 32, 64)
x = inception_module(x, 192, 96, 192, 24, 64, 64)

# MaxPooling and more Inception Modules
x = layers.MaxPooling2D((3,3), strides=(2,2))(x)
x = inception_module(x, 192, 96, 192, 24, 64, 64)

# Global Average Pooling and Fully Connected Layer
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.4)(x)

# Output Layer (3 classes, Shady, Roof, Solarpanel)
output_layer = layers.Dense(3, activation='softmax')(x)

# Define the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()



#Inception V3 using keras ---- testing Required 

'''
# Import required libraries
import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Load the InceptionV3 model pre-trained on ImageNet without the top fully connected layers
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Freeze the base model (optional, allows fine-tuning later)
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(6, activation='softmax')(x)  # Change 6 to the number of classes in your dataset

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Display the model summary
model.summary()

# Prepare data generators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/final_images',  # Replace with your training data path
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    'path_to_validation_data',  # Replace with your validation data path
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10  # Adjust the number of epochs based on your dataset
)

# Fine-tuning (optional, for further improvement)
# Unfreeze some layers in the base model and continue training
for layer in base_model.layers[:249]:
    layer.trainable = False
for layer in base_model.layers[249:]:
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training with fine-tuning
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10  # Additional fine-tuning epochs
)
'''