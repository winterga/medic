import os
import argparse
from PIL import Image

def resize_image(input_image, size=(512, 512)):
    """
    Resize the given image to the specified size and overwrite the original image.
    
    :param input_image: Path to the image file
    :param size: Tuple of (width, height) to resize the image to
    """
    if input_image.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
        try:
            with Image.open(input_image) as img:
                img = img.resize(size, Image.LANCZOS)  # Resize image
                img.save(input_image)  # Overwrite original image
                print(f'Resized and overwritten: {input_image}')
        except Exception as e:
            print(f'Error processing {input_image}: {e}')

def main():
    parser = argparse.ArgumentParser(description='Resize a single image to 512x512.')
    parser.add_argument('input_image', type=str, help='Path to the image file to resize.')
    
    args = parser.parse_args()
    resize_image(args.input_image)

if __name__ == "__main__":
    main()
