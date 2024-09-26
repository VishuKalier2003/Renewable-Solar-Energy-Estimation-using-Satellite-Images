import tensorflow as tf
from tensorflow.keras import layers, Model

# Vision Transformer (ViT) implementation

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

def vision_transformer(input_shape, num_classes, patch_size, num_patches, projection_dim, num_heads, transformer_layers, mlp_units):
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # Patch embedding
    patches = layers.Conv2D(projection_dim, kernel_size=patch_size, strides=patch_size)(inputs)
    patches = layers.Reshape((num_patches, projection_dim))(patches)

    # Positional embedding
    positions = tf.range(start=0, limit=num_patches, delta=1)
    positional_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded_patches = patches + positional_embedding

    # Transformer layers
    for _ in range(transformer_layers):
        # Layer normalization 1
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(x1, x1)
        # Skip connection 1
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP
        x3 = mlp(x3, hidden_units=mlp_units, dropout_rate=0.1)
        # Skip connection 2
        encoded_patches = layers.Add()([x3, x2])

    # Global average pooling
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)

    # Classification head
    logits = layers.Dense(num_classes)(representation)

    # Define the model
    model = Model(inputs=inputs, outputs=logits)
    return model

# Model parameters
input_shape = (224, 224, 3)  # Modify based on image size and channels
num_classes = 2  # Solar panel present or not (binary classification)
patch_size = 16
num_patches = (input_shape[0] // patch_size) * (input_shape[1] // patch_size)
projection_dim = 64
num_heads = 4
transformer_layers = 8
mlp_units = [128, 64]

# Build the Vision Transformer model
vit_model = vision_transformer(input_shape, num_classes, patch_size, num_patches, projection_dim, num_heads, transformer_layers, mlp_units)

# Compile the model
vit_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Model Summary
vit_model.summary()
