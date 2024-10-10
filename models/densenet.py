import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore

# Dense block where each layer is connected to every other layer
def dense_block(x, num_layers, growth_rate):
    for i in range(num_layers):
        # Batch Normalization
        bn = layers.BatchNormalization()(x)
        
        # ReLU activation
        relu = layers.Activation('relu')(bn)
        
        # 3x3 Convolution
        conv = layers.Conv2D(growth_rate, (3, 3), padding='same')(relu)
        
        # Concatenate the input and output of the layer (Dense connectivity)
        x = layers.Concatenate()([x, conv])
    
    return x

# Transition block to reduce the feature map size and apply compression
def transition_layer(x, compression_factor):
    # Batch Normalization
    bn = layers.BatchNormalization()(x)
    
    # ReLU activation
    relu = layers.Activation('relu')(bn)
    
    # 1x1 Convolution for compression
    num_filters = int(x.shape[-1] * compression_factor)
    conv = layers.Conv2D(num_filters, (1, 1), padding='same')(relu)
    
    # Average Pooling for downsampling
    x = layers.AveragePooling2D((2, 2), strides=(2, 2))(conv)
    
    return x

# DenseNet Model Definition
def DenseNet(input_shape, num_classes, num_dense_blocks=4, num_layers_per_block=4, growth_rate=12, compression_factor=0.5):
    inputs = layers.Input(shape=input_shape)

    # Initial Convolutional layer
    x = layers.Conv2D(64, (7, 7), padding='same', strides=(2, 2))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Adding Dense Blocks followed by Transition layers
    for i in range(num_dense_blocks):
        # Dense Block
        x = dense_block(x, num_layers_per_block, growth_rate)

        # Transition layer (except after the last dense block)
        if i != num_dense_blocks - 1:
            x = transition_layer(x, compression_factor)

    # Final Batch Normalization
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Global Average Pooling to reduce dimensionality
    x = layers.GlobalAveragePooling2D()(x)

    # Fully Connected Layer (Classifier)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    # Define the model
    model = Model(inputs, outputs)

    return model

# Define model parameters
input_shape = (640, 640, 3)  # ImageNet-like input shape (224x224 RGB images)
num_classes = 1  # Number of output classes (e.g., solar panels, Shade, Roof.)
growth_rate = 12  # Growth rate of feature maps in each dense block
compression_factor = 0.5  # Compression factor for the transition layers
num_dense_blocks = 4  # Number of dense blocks
num_layers_per_block = 6  # Number of layers per dense block

# Build the DenseNet model
densenet_model = DenseNet(input_shape, num_classes, num_dense_blocks, num_layers_per_block, growth_rate, compression_factor)

# Compile the model
densenet_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# Model summary
densenet_model.summary()


'''
to be added into the train.py for model training... fit()
densenet_model.fit(train_data, train_labels, epochs=50, batch_size=32, validation_data=(val_data, val_labels))
'''

