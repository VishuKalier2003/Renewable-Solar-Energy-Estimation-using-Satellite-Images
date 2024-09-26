# HED Processed Images

## Description

This folder contains images that have been processed using the Holistically-nested Edge Detection (HED) algorithm. These images represent an important stage in our image processing pipeline.

## Contents

The images in this folder have the following characteristics:

- They are the result of applying the HED algorithm to blue-filtered images.
- File names typically end with "_hed.png" to indicate HED processing.

## Previous Step

Before HED processing, images were stored in the `blue_filtered_images` folder after having a blue filter applied.

## Next Steps

After HED processing, these images will undergo final processing and be moved to the `final_images` folder.

## Usage

1. Do not manually add or remove images from this folder unless you're certain of what you're doing.
2. The HED processing is typically handled automatically by our image processing scripts.
3. To move these images to final processing, run the appropriate script (refer to the main project documentation for details).

## File Naming Convention

- Blue filtered image (from previous step): `example_blue.png`
- HED processed image (in this folder): `example_blue_hed.png`

## Notes

- HED processing can be computationally intensive. Ensure your system meets the required specifications.
- The HED algorithm highlights edges and structures in the images, which is crucial for subsequent analysis.
- If you notice any anomalies in the HED-processed images, check the original and blue-filtered versions for potential issues.

## Related Folders

- Blue Filtered Images: `../blue_filtered_images`
- Final Images: `../final_images`

For more information on the HED algorithm and its role in the overall image processing workflow, please refer to the main project README and documentation.
