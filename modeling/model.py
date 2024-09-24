'''
This script loads the images of the validation set and the model we have trained and it will predict the labels of the images.
The script prints out the number of images in each class predicted by the model.
'''

### Import packages

import os
import pandas as pd

import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
import torchvision.transforms.v2 as transforms
import tqdm
from PIL import Image


### Import images

# Convert 'value' column to appropriate data types
def convert_value(value):
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value  # Return as string if it can't be converted to number

param_path = '../data/temp/parameters.csv'

if os.path.exists(param_path):
    df_param = pd.read_csv(param_path)
    df_param.set_index('parameter', inplace=True)
    # Apply the conversion function to the 'value' column
    df_param['value'] = df_param['value'].apply(convert_value)
    param = df_param.to_dict()['value']

else:
    param = {}

    # if we want to try one part of our dataset
    param['frac'] = 0.1 
    param['random_state'] = 1 
    
    param['data_folder'] = '../data' # if this file and folders train_features and test_features are in the same folder use '.'
    
    param['label_csv'] = os.path.join(param['data_folder'], 'train_labels.csv')
    param['train_features_csv'] = os.path.join(param['data_folder'], 'train_features.csv')
    
    param['test_features_csv'] = os.path.join(param['data_folder'], 'test_features.csv')
    param['results_csv_path'] = os.path.join(param['data_folder'], 'test_predictions.csv')
    

    # Convert dictionary to DataFrame with parameters as index and save to csv
    df_param = pd.DataFrame(list(param.items()), columns=['parameter', 'value']).set_index('parameter')



# Create variables and assign values
for key, value in param.items():
    globals()[key] = value
#print(df_param)

if os.path.exists(param_path):
    # if we already split our dataset to train and validation parts, we use these dataframes only to see the data (only train part of train_features)
    train_labels = pd.read_csv(output_train_label_csv, index_col="id") 
    train_features = pd.read_csv(output_train_features_csv, index_col="id")
    test_features = pd.read_csv(output_test_features_csv, index_col="id")

    # we are creating here our train and validation datasets for our model because we have them preprocessed
    x_train = pd.read_csv(output_train_features_csv, index_col="id").filepath.to_frame()
    x_eval = pd.read_csv(output_val_features_csv, index_col="id").filepath.to_frame()
    y_train = pd.read_csv(output_train_label_csv, index_col="id")
    y_eval = pd.read_csv(output_val_label_csv, index_col="id")

else:
    # if we did nothing in preprocessing.ipynb and don't have any preprocessed pictures and any file of parameters, we are taking initial pictures and will split them later
    train_labels = pd.read_csv(label_csv, index_col="id")
    train_features = pd.read_csv(train_features_csv, index_col="id")
    test_features = pd.read_csv(test_features_csv, index_col="id")

    train_features['filepath'] = data_folder+'/'+train_features.filepath
    test_features['filepath'] = data_folder+'/'+test_features.filepath


species_labels = ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']

## class for importing a dataset of images

class ImagesDataset(Dataset):
    """Reads in an image, transforms pixel values, and serves
    a dictionary containing the image id, image tensors, and label.
    """

    def __init__(self, x_df, y_df=None):
        self.data = x_df
        self.label = y_df
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __getitem__(self, index):
        image = Image.open(self.data.iloc[index]["filepath"]).convert("RGB")
        image = self.transform(image)
        image_id = self.data.index[index]
        # if we don't have labels (e.g. for test set) just return the image and image id
        if self.label is None:
            sample = {"image_id": image_id, "image": image}
        else:
            label = torch.tensor(self.label.iloc[index].values, 
                                 dtype=torch.float)
            sample = {"image_id": image_id, "image": image, "label": label}
        return sample

    def __len__(self):
        return len(self.data)


## create DataLoader object
eval_dataset = ImagesDataset(x_eval, y_eval)
eval_dataloader = DataLoader(eval_dataset, batch_size=32)

### load model and make prediction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("../models/model.pth")
model.to(device)
preds_collector = []

# put the model in eval mode so we don't update any parameters
model.eval()
# We aren't updating our weights so no need to calculate gradients
with torch.no_grad():
    for batch in tqdm.tqdm(eval_dataloader, total=len(eval_dataloader)):
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
eval_preds_df = pd.concat(preds_collector)                      # this datafraims contains the probabilites for each class to be true
# Display the dataframe (optional)
eval_preds_df

# remove the model/images from GPU and clear GPU cache
model.cpu()
images.cpu()
torch.cuda.empty_cache()

# Here we count the predicted number of instances for each class
eval_predictions = eval_preds_df.idxmax(axis=1)
prediction_count = eval_predictions.value_counts()

### return predicted labels
print(prediction_count)