# Model Structures

## Description

This folder contains the architectural definitions of various machine learning models used in our project. These model structures are designed to be imported and used by the main training script (main.py) for different tasks related to solar energy estimation and satellite image analysis.

## Contents

This folder should contain Python files, each defining a specific model architecture. For example:

- `custom-yolo.py`
- `densenet.py`
- `inception.py`
- `vision_transformer.py`

## Purpose

The purpose of this folder is to:

1. Centralize all model definitions in one location
2. Allow easy importing and use of models in the main training script
3. Facilitate quick comparisons and experimentations with different model architectures

## Usage

1. Defining New Models:
   - Create a new Python file for each new model architecture.
   - Use clear, descriptive names for both the file and the model class.
   - Include docstrings and comments to explain the model's structure and purpose.

2. Importing Models:
   - In main.py, import models as needed:

     ```python
     from model_structures.cnn_model import CNNModel
     from model_structures.unet_model import UNetModel
     ```

3. Modifying Existing Models:
   - When modifying an existing model, consider creating a new version rather than overwriting.
   - Clearly document changes in the file and update any relevant documentation.

## File Structure

Each model file should follow a similar structure:

```python
import tensorflow as tf  # or your preferred ML framework

class ModelName(tf.keras.Model):
    def __init__(self, params):
        super(ModelName, self).__init__()
        # Define layers and model structure

    def call(self, inputs):
        # Define forward pass
        return outputs
```

Documentation: Include detailed docstrings explaining inputs, outputs, and key architectural decisions.

## Notes

- Ensure all required dependencies are listed in the project's main requirements.txt file.
- Keep model definitions modular and reusable where possible.
- Consider adding a brief comment at the top of each file indicating which task(s) the model is designed for.

## Related Files

- main.py: The main script that imports and uses these model structures
- train.py: (if separate from main.py) The training script that utilizes these models
- config.py: Configuration file that might specify which model to use

## Contribution Guidelines

When adding or modifying models:

1. Follow the existing code style and structure.
2. Update this README if you add new model types or change usage patterns.
3. Ensure your changes don't break existing functionality in train.py.
4. Add appropriate tests for new models or modifications.

This README provides a comprehensive guide for managing and using the model structures in your project. It covers the purpose of the folder, how to use and contribute to it, best practices, and important considerations for maintaining and expanding your collection of model architectures.
