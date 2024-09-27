import os
import importlib
import argparse

def train_model(model_name):
    # Import the model module dynamically
    model_module = importlib.import_module(f"model.{model_name}")
    model_class = getattr(model_module, model_name.capitalize())
    
    # Load the model instance
    model = model_class()
    
    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    
    # Load the training data
    # Replace this with your actual data loading code
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Train the model
    model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))
    
    # Save the weights
    weights_file = f"{model_name}.{get_weights_extension()}"
    weights_path = os.path.join("weights", weights_file)
    model.save_weights(weights_path)

def get_weights_extension():
    # You can modify this function to return the appropriate weights file extension
    # For example, if you're using TensorFlow, you might return ".ckpt"
    # If you're using PyTorch, you might return ".pth"
    return ".h5"  # Default to H5 format

def main():
    # Get a list of available models
    model_names = [f[:-3] for f in os.listdir("model") if f.endswith(".py")]
    
    # Train each model
    for model_name in model_names:
        print(f"Training model {model_name}...")
        train_model(model_name)
        print(f"Model {model_name} trained and weights saved.")

if __name__ == "__main__":
    main()
