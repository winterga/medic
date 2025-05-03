import cv2
import os
import argparse
import numpy as np

def crop_non_black_area(image, threshold=0, padding=10):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where non-black pixels are white
    _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of the non-black areas
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return None
    if not contours:
        print("Error: No non-black areas found.")
        return None

    # Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Get the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Apply padding to the bounding box
    x = max(x - padding, 0)  # Ensure it doesn't go out of bounds
    y = max(y - padding, 0)  # Ensure it doesn't go out of bounds
    w = min(w + 2 * padding, image.shape[1] - x)  # Ensure it doesn't go out of bounds
    h = min(h + 2 * padding, image.shape[0] - y)  # Ensure it doesn't go out of bounds

    # Crop the image to the padded bounding box
    cropped_image = image[y:y + h, x:x + w]

    # Check if the cropped image is empty
    if cropped_image.size == 0:
        print("Error: Cropped image is empty.")
        return None

    return cropped_image



def crop_and_resize(image, threshold=0, padding=10, target_size=(512, 512)):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No non-black areas found.")
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LANCZOS4)

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    x = max(x - padding, 0)
    y = max(y - padding, 0)
    w = min(w + 2 * padding, image.shape[1] - x)
    h = min(h + 2 * padding, image.shape[0] - y)

    cropped_image = image[y:y + h, x:x + w].copy()
    cropped_image = np.asarray(cropped_image, dtype=image.dtype)
    resized_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_LANCZOS4)

    return resized_image




def process_single_image(input_image, threshold=0, padding=10):
    # Check if the file is an image
    if input_image.lower().endswith(('.jpg', '.jpeg', '.png')):
        # Read the image
        image = cv2.imread(input_image)
        if image is None:
            print(f"Error: Unable to read image at {input_image}.")
            return

        # Crop the non-black area
        cropped_image = crop_non_black_area(image, threshold, padding)

        # If cropping was successful, save the new image, replacing the old one
        if cropped_image is not None:
            cv2.imwrite(input_image, cropped_image)
            # print(f"Cropped image saved to {input_image}.")
        else:
            print(f"Skipping {input_image} due to cropping error.")
    else:
        print(f"File {input_image} is not a supported image format.")

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crop non-black areas from a single image.")
    parser.add_argument('input_image', type=str, help='Path to the image to process')
    parser.add_argument('--threshold', type=int, default=0, help='Threshold for non-black pixels')
    parser.add_argument('--padding', type=int, default=0, help='Padding around the cropped area')

    args = parser.parse_args()

    process_single_image(args.input_image, threshold=args.threshold, padding=args.padding)
