# Document Layout Mask Generator

This tool creates layout masks for document images using OpenCV's Otsu thresholding method. It processes document images to generate binary masks that separate text regions from the background, along with corresponding JSON files containing bounding box coordinates.

## Features

- Processes multiple image formats (PNG, JPG, JPEG)
- Generates binary masks for text regions
- Creates JSON files with bounding box coordinates
- Handles large images with PIL's decompression bomb limit removed
- Progress bar visualization using tqdm

## Requirements

```python
pip install opencv-python
pip install numpy
pip install Pillow
pip install tqdm
pip install torch
```

## Usage

Run the script from command line:

```bash
python create_layout_masks.py --image_dir /path/to/images --output_dir /path/to/output
```

### Arguments

- `--image_dir`: Directory containing the input document images
- `--output_dir`: Directory where masks and JSON files will be saved

### Output

For each input image, the script generates:

1. A binary mask file (`*_mask.png`) where:
   - White pixels (255) represent text regions
   - Black pixels (0) represent background
2. A JSON file containing:
   - Original image path
   - List of bounding boxes for detected text regions

## How It Works

1. Loads each image and converts it to grayscale
2. Applies Gaussian blur for noise reduction
3. Uses Otsu's thresholding to separate text from background
4. Finds contours and filters them by size
5. Creates binary masks and extracts bounding boxes
6. Saves both mask images and JSON metadata

## Notes

- Minimum text region size filters: width > 50px, height > 20px
- Images are processed in RGB format
- The script automatically creates the output directory if it doesn't exist

## License

[Add your license information here]
