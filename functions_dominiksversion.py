import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
import random
import os
import shutil
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms import functional as F  # Import torchvision.transforms.functional

from torch.utils.data import DataLoader
from torch import nn
import torchvision.models as models
import torch.optim as optim


def top_3_predictions_with_labels(submission_df):
    """
    Create a DataFrame with the top 3 maximum probabilities and their corresponding labels.
    Rows are sorted by 'top1' in descending order.
    
    Args:
        submission_df (pd.DataFrame): Input DataFrame with probabilities for each label.
        
    Returns:
        pd.DataFrame: A new DataFrame with columns 'top1', 'top2', 'top3' and 
                      'top1_label', 'top2_label', 'top3_label', sorted by 'top1'.
    """
    # Create lists to store the top 3 probabilities and their corresponding labels
    top1, top2, top3 = [], [], []
    top1_labels, top2_labels, top3_labels = [], [], []

    for index, row in submission_df.iterrows():
        # Get the top 3 probabilities and their corresponding labels
        sorted_probs = row.sort_values(ascending=False)[:3]
        
        # Append the top probabilities and labels
        top1.append(sorted_probs.iloc[0])
        top2.append(sorted_probs.iloc[1])
        top3.append(sorted_probs.iloc[2])
        top1_labels.append(sorted_probs.index[0])
        top2_labels.append(sorted_probs.index[1])
        top3_labels.append(sorted_probs.index[2])
    
    # Create a new DataFrame with the results
    top_3_df = pd.DataFrame({
        'top1': top1,
        'top2': top2,
        'top3': top3,
        'top1_label': top1_labels,
        'top2_label': top2_labels,
        'top3_label': top3_labels
    }, index=submission_df.index)
    
    # Sort by 'top1' in descending order
    top_3_df = top_3_df.sort_values(by='top1', ascending=False)
    
    return top_3_df


def handle_model_folder(model_folder):
    """
    Function to either delete the contents of an existing model folder and create a new one, 
    or rename the existing folder and create a new one with the original name.
    """
    # Check if the folder exists
    if os.path.exists(model_folder):
        # Ask the user what to do with the existing folder
        action = input(f"The folder {model_folder} already exists. What would you like to do?\n"
                       "Enter 'd' to delete its contents or 'r' to rename the folder: ").lower()

        if action == 'd':
            # Delete the directory along with its contents
            shutil.rmtree(model_folder)
            print(f"The folder {model_folder} has been deleted.")
            # Create a new empty folder with the same name
            os.makedirs(model_folder)
            print(f"New empty folder created: {model_folder}")

        elif action == 'r':
            # Construct the default new name for the folder
            parent_dir = os.path.dirname(model_folder)
            default_new_name = os.path.join(parent_dir, os.path.basename(model_folder) + '_old')

            # Ask for the new name, pre-filling with the default name
            new_name = input(f"Enter the new name for the folder (default: {default_new_name}): ") or default_new_name

            # Ensure the new name is different from the original
            while os.path.exists(new_name):
                new_name = input(f"A folder named {new_name} already exists. Please enter a different name: ")

            # Rename the folder
            os.rename(model_folder, os.path.join(parent_dir, new_name))
            print(f"The folder {model_folder} has been renamed to {new_name}.")
            # Create a new empty folder with the original name
            os.makedirs(model_folder)
            print(f"New empty folder created: {model_folder}")

    else:
        # Create a new empty folder if it doesn't exist
        os.makedirs(model_folder)
        print(f"New empty folder created: {model_folder}")


def convert_value(value):
    """
    Converts a value to its appropriate data type: boolean, integer, float, or string.

    Args:
        value (str, int, float): The value to be converted.

    Returns:
        bool, int, float, or str: The converted value. Returns True/False for boolean 
                                  strings, integer/float for numeric strings, or 
                                  the original string if no conversion is possible.
    """
    # First, check for boolean string representations
    if isinstance(value, str):
        lower_value = value.lower()
        if lower_value == 'true':
            return True
        elif lower_value == 'false':
            return False

    # Then, try to convert the value to int or float
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Return as string if it can't be converted to a number or boolean

        
def get_pth_file(folder):
    """
    Function to find and return the .pth file in the given folder.
    Returns the full path to the .pth file if found, or None if not.
    """
    for file_name in os.listdir(folder):
        if file_name.endswith('.pth'):
            return os.path.join(folder, file_name)
    return None


def load_csv_if_exists(filename, use_index=False):
    """
    Load a CSV file if it exists.
    Optionally use the first column as the DataFrame index.
    
    Args:
        filename (str): Path to the CSV file.
        use_index (bool): Whether to use the first column as the index (default: False).
        
    Returns:
        pd.DataFrame or None: Loaded DataFrame or None if the file does not exist.
    """
    if os.path.exists(filename):
        if use_index:
            # Load the CSV with the first column as the index
            return pd.read_csv(filename, index_col=0)
        else:
            # Load the CSV without using the first column as the index
            return pd.read_csv(filename)
    else:
        print(f"File '{filename}' does not exist.")
        return None  # Or alternatively return pd.DataFrame() if you prefer an empty DataFrame


def df_to_dict(df):
    """
    Converts a DataFrame to a dictionary where keys are tuples of the first two columns
    and the values are from the third column.

    Args:
        df (pd.DataFrame): Input DataFrame with at least three columns.

    Returns:
        dict: A dictionary where keys are tuples of (first_column, second_column),
              and values are from the third column.
    """
    result_dict = {}
    for _, row in df.iterrows():
        key = (int(row.iloc[0]), int(row.iloc[1]))  # Use iloc to access by position
        value = float(row.iloc[2])  # Get the value from the third column
        result_dict[key] = value
    return result_dict


def create_image_dataset(image_folder):
    """
    Create a DataFrame where 'id' column is used as the index (without .jpg extension)
    and 'filepath' column contains the path for each image.
    The index is sorted in ascending order.
    Checks if the folder exists before proceeding.
    """
    # Check if the folder exists
    if not os.path.exists(image_folder):
        print(f"Folder '{image_folder}' does not exist.")
        return pd.DataFrame()  # Return an empty DataFrame if folder doesn't exist
    
    # Get all .jpg files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    
    # Create a list of dictionaries with 'id' (filename without .jpg) and 'filepath'
    data = [{'id': os.path.splitext(file)[0], 'filepath': os.path.join(image_folder, file)} for file in image_files]
    
    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    
    # Set 'id' as the index and sort the index in ascending order
    df = df.set_index('id').sort_index()
    
    return df


# Define the inverse normalization transform
inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225], 
    std=[1/0.229, 1/0.224, 1/0.225]
)

def display_random_images(dataset, num_images=12):
    # Randomly sample `num_images` indices from the dataset
    random_indices = random.sample(range(len(dataset.data)), num_images)
    
    plt.figure(figsize=(15, 10))

    for i, idx in enumerate(random_indices):
        # Get the sample at the given index
        sample = dataset[idx]
        
        # Convert the image tensor back to its original range (0 to 1)
        image_tensor = sample['image']
        
        # Apply inverse normalization
        image_tensor = inv_normalize(image_tensor)

        # Convert the normalized tensor to a PIL image for display
        image = transforms.ToPILImage()(image_tensor)

        plt.subplot(3, 4, i + 1)
        plt.imshow(image)
        plt.title(f"Image ID: {sample['image_id']}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


def make_predictions(checked_dataloader,model,species_labels):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    preds_collector = []

    # put the model in eval mode so we don't update any parameters
    model.eval()

    # We aren't updating our weights so no need to calculate gradients
    with torch.no_grad():
        for batch in tqdm(checked_dataloader, total=len(checked_dataloader)):
            
            # Move batch data to the device
            images = batch["image"].to(device)
            image_ids = batch["image_id"]
            # 1) Run the forward step
            logits = model(images)
            # 2) Apply softmax so that model outputs are in range [0,1]
            preds = nn.functional.softmax(logits, dim=1)
            # 3) Store this batch's predictions in df
            # Ensure tensors are moved to CPU and detached from computational graph before converting to numpy arrays
            preds_df = pd.DataFrame(
                preds.cpu().detach().numpy(),
                index=image_ids,
                columns=species_labels,
            )
            preds_collector.append(preds_df)

    # Concatenate all dataframes into a single dataframe
    eval_preds_df = pd.concat(preds_collector)
    # Display the dataframe (optional)
    eval_preds_df = eval_preds_df.sort_index()

    model.cpu()
    images.cpu()
    torch.cuda.empty_cache()

    return eval_preds_df


class ImagesDataset(Dataset):
    def __init__(self, x_df, y_df=None, use_padding=False, use_white_line_cropping=True, use_logo_cropping=True, channels_number=3):
        self.data = x_df
        self.label = y_df
        self.use_padding = use_padding
        self.use_white_line_cropping = use_white_line_cropping
        self.use_logo_cropping = use_logo_cropping
        self.channels_number = channels_number

    def crop_white_bottom_from_image(self, img, check_height=8, check_width=10, start_from=60, threshold=230):
        width, height = img.size
        box = (start_from, height - check_height, start_from + check_width, height)
        region = img.crop(box)
        region_np = np.array(region)
        white_pixels = np.apply_along_axis(lambda color: all(c >= threshold for c in color), 1, region_np)

        if np.all(white_pixels):
            img = img.crop((0, 0, width, height - check_height - 8))
        return img

    def is_bright_orange(self, rgb):
        r, g, b = rgb
        # Approximate boundaries for bright orange color in RGB
        return (r > 180 and 50 < g < 200 and b < 110)

    def replace_logo_with_neighboring_colors(self, img, search_area_height=50, search_area_width=50):
        width, height = img.size
        pixels = img.load()

        # List to store coordinates of orange pixels
        orange_points = []

        # Iterate over the bottom-left 50x50 rectangle
        for x in range(search_area_width):
            for y in range(height - search_area_height, height):
                if self.is_bright_orange(pixels[x, y]):
                    orange_points.append((x, y))

        # If orange pixels are found
        if orange_points:
            # Find the top-right orange point and take some points more
            max_orange_x = max(x for x, y in orange_points) + 3
            min_orange_y = min(y for x, y in orange_points) - 5

            # Replace logo with vertical lines. Color is like color of pixel above
            for x in range(max_orange_x):
                color = pixels[x, min_orange_y - 1]
                for y in range(min_orange_y, height):
                    pixels[x, y] = color
        return img

    def resize_and_normalize_image(self, image):
        """Resize and normalize the image based on the `use_padding` parameter."""
        if not self.use_padding:
            # Resize directly to 224x224 without keeping aspect ratio
            image = F.resize(image, (224, 224))
        else:
            # Resize with padding to ensure the larger side is 224
            max_dim = max(image.size)
            scale = 224 / max_dim
            new_size = (int(image.height * scale), int(image.width * scale))
            image = F.resize(image, new_size)

            # Create a new black image and paste the resized image onto the center
            new_size = image.size  # Size of the resized image

            # Create a new black image (224x224)
            new_image = Image.new("RGB", (224, 224))

            # Calculate the correct paste position to center the image
            paste_position = ((224 - new_size[0]) // 2, (224 - new_size[1]) // 2)

            # Paste the resized image onto the center of the black square
            new_image.paste(image, paste_position)

            # Update the image to the new centered image
            image = new_image
            

        # Convert to tensor and normalize
        image_tensor = transforms.ToTensor()(image)
        normalized_image = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(image_tensor)
        return normalized_image

    def convert_to_grayscale_rgb(self, image):
        """Convert the image to grayscale and then back to RGB format."""
        grayscale_image = ImageOps.grayscale(image)  # Convert to grayscale
        rgb_image = ImageOps.colorize(grayscale_image, black="black", white="white")  # Back to RGB with neutral colors
        return rgb_image

    def __getitem__(self, index):
        if index >= len(self.data):
            raise IndexError("Index out of range")

        image_path = self.data.loc[self.data.index[index], "filepath"]
        image = Image.open(image_path).convert("RGB")

        # Apply white line cropping if enabled
        if self.use_white_line_cropping:
            image = self.crop_white_bottom_from_image(image)

        # Apply logo removal if enabled
        if self.use_logo_cropping:
            image = self.replace_logo_with_neighboring_colors(image)

        # Convert to grayscale and back to RGB if channels_number is set to 1
        if self.channels_number == 1:
            image = self.convert_to_grayscale_rgb(image)

        normalized_image = self.resize_and_normalize_image(image)
        image_id = self.data.index[index]

        sample = {"image_id": image_id, "image": normalized_image}

        if self.label is not None:
            label_index = self.data.index[index]
            if label_index in self.label.index:
                label = torch.tensor(self.label.loc[label_index].values, dtype=torch.float)
                sample["label"] = label

        return sample

    def __len__(self):
        return len(self.data)



    

