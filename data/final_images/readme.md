# Final Images Folder

## Description

This folder contains the fully processed and prepared images that are ready for use in training our machine learning models. These images have undergone all necessary preprocessing steps and represent the final stage of our image preparation pipeline.

## Contents

- Processed satellite images or other relevant image data
- Consistent image format (e.g., .png)
- Standardized size and resolution
- Normalized and enhanced images

## Purpose

The images in this folder serve as the direct input for training various machine learning models in our project. They have been carefully processed to ensure optimal quality and consistency for model training.

## Preprocessing Steps Completed

Images in this folder have undergone the following processing steps:

1. Initial preprocessing (resizing, normalization)
2. Blue filtering
3. Holistically-nested Edge Detection (HED)
4. Final adjustments (if any)

## Usage Guidelines

1. Using Images for Training:
   - These images are ready to be fed directly into your model training pipelines.
   - Ensure your data loading scripts are pointing to this folder.

2. Adding or Removing Images:
   - Do not manually add or remove images from this folder.
   - All additions should come through the automated preprocessing pipeline.
   - If removal is necessary, ensure it's reflected in your training data manifests.

3. Versioning:
   - Consider versioning this folder if significant changes are made to the preprocessing pipeline.
   - You might use subfolders like `v1/`, `v2/`, etc., for different versions of processed data.

## File Naming Convention

- Files should follow a consistent naming pattern: `original_name_final.png`
- Example: `20230615_california_coastal_region1_final.png`

## Notes

- Ensure all models and team members are using the same version of processed images.
- Regularly backup this folder to prevent data loss.
- Document any changes to the preprocessing pipeline that affects these images.

## Model Training

Images from this folder will be used to train various models, including but not limited to:

- Solar energy potential estimation
- Land use classification
- Environmental change detection

Ensure that your model training scripts are correctly configured to access and utilize these images.

## Related Folders

- Original Input Images: `../input_images/`
- Blue Filtered Images: `../blue_filtered_images/`
- HED Processed Images: `../output_hed_images/`

## Data Validation

Before using these images for training:

1. Verify that all expected images are present.
2. Conduct random spot checks to ensure image quality.
3. Run automated checks for consistency in size, format, and other relevant attributes.

## Related Documentation

For more information on how these images are used in model training, please refer to:

- Main project README
- Model training documentation
- Data pipeline documentation

If you encounter any issues with the images or need to reprocess the dataset, please contact the project's data processing lead or machine learning engineer.
