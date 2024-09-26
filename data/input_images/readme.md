# Input Images Folder

## Description

This folder contains the raw, unprocessed images that serve as the starting point for our image processing and machine learning pipeline. These images will undergo various preprocessing steps before being used for training our models.

## Contents

- Raw satellite images or other relevant image data
- Various image formats (e.g., .png, .jpg, .tiff)
- Potentially large file sizes

## Purpose

The images in this folder are the foundation of our project. They will be processed through several stages to prepare them for model training, including:

1. Initial preprocessing (e.g., resizing, normalization)
2. Blue filtering
3. Holistically-nested Edge Detection (HED)
4. Final preprocessing

## Usage Guidelines

1. Adding Images:
   - Place new, unprocessed images directly into this folder.
   - Ensure images are in supported formats (.png, .jpg, .tiff, etc.).
   - Maintain consistent naming conventions if possible.

2. Removing Images:
   - Only remove images if they are corrupted or no longer needed for the project.
   - Ensure you have backups before deleting any files.

3. Organizing:
   - You may create subfolders to organize images by date, location, or other relevant categories.

4. Processing:
   - Do not manually edit or process images in this folder.
   - Use the project's preprocessing scripts to handle these images.

## File Naming Convention

- Try to use descriptive, consistent names: `YYYYMMDD_location_details.format`
- Example: `20230615_california_coastal_region1.png`

## Notes

- Ensure you have sufficient disk space, as this folder may contain large image files.
- Regularly backup this folder to prevent data loss.
- If adding new image types, ensure the preprocessing scripts can handle the new format.

## Next Steps in Pipeline

Images from this folder will be processed and moved through the following stages:

1. `blue_filtered_images/`
2. `output_hed_images/`
3. `final_images/`

## Related Documentation

For more information on the preprocessing steps and overall workflow, please refer to:

- Main project README
- Preprocessing documentation
- Data pipeline documentation

If you have any questions about adding or managing images in this folder, please contact the devs.
