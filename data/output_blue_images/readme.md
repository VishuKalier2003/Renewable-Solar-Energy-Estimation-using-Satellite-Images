# Blue Filtered Images

## Description

This folder contains images that have been processed with a blue filter. These images serve as an intermediate step in our image processing pipeline.

## Contents

The images in this folder have the following characteristics:

- They are the result of applying a blue filter to original input images.
- File names typically end with "_blue.png" to indicate the blue filter processing.

## Next Steps

After the blue filter processing, these images will be further processed using the Holistically-nested Edge Detection (HED) algorithm. The results of the HED processing will be stored in the `output_hed_images` folder.

## Usage

1. Do not manually add or remove images from this folder unless you're sure of what you're doing.
2. The blue filter processing is typically handled automatically by our image processing scripts.
3. To process these images with HED, run the appropriate script (refer to the main project documentation for details).

## File Naming Convention

- Original image: `example.png`
- Blue filtered image (in this folder): `example_blue.png`

## Notes

- Ensure you have sufficient disk space, as image processing can generate large files.
- If you encounter any issues with the images in this folder, please refer to the troubleshooting section in the main project documentation.

## Related Folders

- Input Images: `../input_images`
- HED Processed Images: `../output_hed_images`

For more information on the overall image processing workflow, please refer to the main project README.md
