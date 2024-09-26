# Renewable-Solar-Energy-Estimation-using-Satellite-Images

## Overview

The **Solar-Panel-Detection-from-Spatial-Data** project aims to leverage deep learning techniques to detect solar panels from spatial and satellite data. The primary objective is to create a model that can take input images (annotated spatial data) and accurately identify and locate solar panels. This project utilizes Roboflow for image annotation and preprocessing.

## Project Structure

- **Data Collection and Annotation**:
  - We use high-resolution spatial data to train our model. The dataset includes images of areas with solar panels and non-solar areas for diversity.
  - **Roboflow** is used for annotation and preprocessing (e.g., auto-orient, resize).
  
- **Deep Learning Model**:
  - A custom deep learning model will be designed using convolutional neural networks (CNNs). The goal is to develop a model that takes spatial images as input and outputs predictions on the presence and location of solar panels.
  
- **Output**:
  - The output will consist of labeled bounding boxes around solar panels in input images, with confidence scores indicating detection accuracy.

## Requirements

To get started with the project, the following tools and libraries are required:

- Python 3.10.9
- TensorFlow or PyTorch (for deep learning)
- Roboflow (for dataset management)
- OpenCV (for image handling)
- NumPy, Pandas (for data manipulation)
- Matplotlib and PIL(for visualization)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/VishuKalier2003/Renewable-Solar-Energy-Estimation-using-Satellite-Images
   ```

2. Create Virtual Envirnment:

   ```bash
   python -m venv sp-env
   ```

3. Activate the Envirnment:

   ```bash
   sp-env\Scripts\activate    
   ```

4. Install the requiremnts:

   ```bash
   pip install -r requirements.txt
   ```

5. Set up the dataset (via Roboflow):

   Download your dataset from Roboflow and transfer it to the repo folder. [[Roboflow Link](https://universe.roboflow.com/ml-projects-osdwj/solar-panel-detection-gm1xz)] .

   ```bash
   mkdir data
   ```

    Dataset is pre-organised into train, test, and validation sets. You can follow the dataset import process from the Roboflow documentation.

## Dataset

The dataset is sourced from satellite imagery and contains annotated images of solar panel locations. It is divided into training, validation, and test sets for model evaluation.

Roboflow data image are annotated to class to find/identify solar panel in spartial images.

- **Training Data**: 70% of the dataset.
- **Validation Data**: 20% of the dataset.
- **Test Data**: 10% of the dataset.

## Model Training

- Configure your training settings in the `config.py` file.
- Run the training script:

  ```bash
  python train.py
  ```

The model will be trained using a deep convolutional neural network, with options to adjust hyperparameters like learning rate, batch size, and epochs.

## Inference

After training, you can perform inference on new spatial images:

```bash
python detect.py --input /path/to/image
```

The output will include bounding boxes around detected solar panels and a confidence score.
At later stages we plan to create an Streamline the process via GUI interface.

## Results

- Evaluation metrics include precision, recall, F1-score, and mAP (mean average precision).
- Visualization of the detection results will be stored in the `results` folder.

## Future Work

- **Model Optimization**: Explore optimization techniques to improve detection accuracy and model efficiency.
- **Real-time Deployment**: Integrate the model into a real-time detection system using an API.
- **Cross-domain Generalization**: Apply the model to different types of satellite imagery.

## Contributions

Project is colectivelt developed by:

- **Virat Srivastava**
- **Vishu Kalier**
- **Durgesh Kumar Singh**
