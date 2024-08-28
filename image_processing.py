# Import necessary libraries
from PIL import Image, ImageOps
import os
import numpy as np


def max_sizes(input_folder):
    """Calculates the maximum width and height of all images in the input folder.

    Args:
        input_folder (str): Path to the folder containing the images.

    Returns:
        tuple: Maximum width and height of the images.
    """
    max_width = 0
    max_height = 0
    # Iterate through all images to find the maximum dimensions
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            with Image.open(file_path) as img:
                width, height = img.size
                if width > max_width:
                    max_width = width
                if height > max_height:
                    max_height = height
    return max_width, max_height


def is_white_color(color, threshold=230):
    """Checks if a color is considered white based on a threshold.

    Args:
        color (tuple): A tuple of RGB values.
        threshold (int): Threshold value for considering a color as white.

    Returns:
        bool: True if color is white, False otherwise.
    """
    # Check if color is close to white within the given threshold
    return all(c >= threshold for c in color)


def crop_white_bottom_from_image(img, height_to_check=8, check_width=10, start_from=60, threshold=230):
    """Crops the bottom part of the image if it is predominantly white.

    Args:
        img (PIL.Image): The image to be processed.
        height_to_check (int): Height in pixels to check for white.
        check_width (int): Width in pixels to check for white.
        start_from (int): X-coordinate to start checking for white.
        threshold (int): Threshold for white color.

    Returns:
        PIL.Image: The cropped image if white was found, otherwise original image.
    """
    width, height = img.size
    # Define the region to check for white color
    box = (start_from, height - height_to_check, start_from + check_width, height)
    region = img.crop(box)
    # Convert the region to a NumPy array for easy processing
    region_np = np.array(region)
    # Check if all pixels in the region are close to white
    white_pixels = np.apply_along_axis(is_white_color, 1, region_np)

    if np.all(white_pixels):
        # Crop the bottom 16 pixels
        img = img.crop((0, 0, width, height - height_to_check - 8))
    return img


def padding_images(img, max_width, max_height):
    """Adds padding to an image to match the specified dimensions.

    Args:
        img (PIL.Image): The image to be padded.
        max_width (int): Desired width of the padded image.
        max_height (int): Desired height of the padded image.

    Returns:
        PIL.Image: The padded image.
    """
    # Calculate padding required to center the image
    delta_width = max_width - img.width
    delta_height = max_height - img.height
    padding = (delta_width // 2, delta_height // 2, delta_width - (delta_width // 2), delta_height - (delta_height // 2))
    # Add padding and create a new image with the desired size
    return ImageOps.expand(img, padding, fill=0)


def format_to_rgb(img):
    """Converts an image to RGB format if it is not already in that format.

    Args:
        img (PIL.Image): The image to be converted.

    Returns:
        PIL.Image: The converted image in RGB format.
    """
    if img.mode != 'RGB':
        # Convert image to RGB
        img = img.convert('RGB')
    return img


def crop_white_bottom_add_padding_all_rgb(input_folder, output_folder):
    """Processes all images in the input folder: crops white bottom, adds padding, and converts to RGB.

    Args:
        input_folder (str): Path to the folder containing the images to process.
        output_folder (str): Path to the folder where processed images will be saved.
    """
    max_width, max_height = max_sizes(input_folder)

    # Create the new folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # Iterate through each image in the source folder
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(input_folder, filename)
            with Image.open(file_path) as img:
                img = crop_white_bottom_from_image(img)
                img = padding_images(img, max_width, max_height)
                img = format_to_rgb(img)
                # Save the processed image to the new folder
                new_file_path = os.path.join(output_folder, filename)
                img.save(new_file_path)
    print(output_folder, "Processing complete.")