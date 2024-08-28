import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as sta
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import time
import random
import pickle
import os
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import math
import Modules

## Load integrated work zone traffic data

data = pd.read_csv('processed data/Work_zone_data_with_speed_incident.csv')
num_of_lanes = pd.read_csv('processed data/num_of_lanes_new.csv')
with open('impacted_work_zone_id.pkl', 'rb') as file:
    # Load the data from the file
    data_loaded = pickle.load(file)
    impact_wzid = data_loaded['impact_wzid']
    high_impact_wzid = data_loaded['high_impact_wzid']

## Data cleaning

data_clean_nearWZ = data_cleaning(data):
data_clean_nearWZ = data_clean_nearWZ.drop(columns=['lanes'])
data_clean_nearWZ = pd.merge(data_clean_nearWZ,num_of_lanes,on = 'tmc_code',how = 'left')
data_clean_nearWZ.loc[data_clean_nearWZ['traffic_lane_closure_count']>3,'traffic_lane_closure_count'] = 3
data_clean_nearWZ.loc[data_clean_nearWZ['traffic_lane_closure_count']>data_clean_nearWZ['lanes'],'lanes'] = data_clean_nearWZ.loc[data_clean_nearWZ['traffic_lane_closure_count']>data_clean_nearWZ['lanes'],'traffic_lane_closure_count'].values
data_clean_nearWZ['lane_closure_rate'] = 0
data_clean_nearWZ['lane_closure_rate'] = data_clean_nearWZ['traffic_lane_closure_count']/data_clean_nearWZ['lanes']
inci_wzID = data_clean_nearWZ[(data_clean_nearWZ['inci_wz'] == 1)]['wz_id'].unique()

## Encode categorical features and split data to train/validate/test sets

data_1 = data_clean_nearWZ.copy()
data_1 = encode_categorical_features(data_1):
valid_wzid, id_train_wz, id_val_wz, id_test_wz, selected_id_train_wz_combine = split_train_val_test_workzones(data_1,impact_wzid,high_impact_wzid,inci_wzID):

## Create Input/Output 2D Space-Time Arrays

# normalize numerical features
columns_to_scale = ['speed','duration','volume','average_speed','miles','distance_to_work_zone','TimeOfDay']
scaler = MinMaxScaler(feature_range=(0, 1))
for column_name in columns_to_scale:
    values_to_scale = data_1[column_name].values.reshape(-1, 1)  # Reshape column values
    scaled_values = scaler.fit_transform(values_to_scale)  # Scale values
    data_1[column_name] = scaled_values.flatten()  # Replace original column with scaled values

# Create 2D space-time arrays for input and output
id_array = np.array(selected_id_train_wz_combine)
selected_elements = valid_wzid[np.isin(valid_wzid, id_array)]
wzids = selected_elements.copy()
train_X1, train_X2, train_Y1 = create_sub_sequence(wzids,7,'train')

id_array = np.array(id_val_wz)
selected_elements = valid_wzid[np.isin(valid_wzid, id_array)]
wzids = selected_elements.copy()
val_X1, val_X2, val_Y1,val_df_all = create_sub_sequence(wzids,7,'val')

id_array = np.array(id_test_wz)
selected_elements = valid_wzid[np.isin(valid_wzid, id_array)]
wzids = selected_elements.copy()
test_X1, test_X2, test_Y1, test_df_all = create_sub_sequence(wzids,7,'test')

print(train_X1.shape, train_X2.shape,train_Y1.shape)
valtest_df_all = pd.concat([val_df_all, test_df_all], axis=0).reset_index(drop=True)
valtest_df_all.to_csv('processed data/valtest_df_all.csv')

## Convert arrays to tensors and create dataloaders

# Extract the relevant portion of the array (features 6 to the last)
relevant_array = train_X2[:, 3:]
# Reshape the array to collapse the batch and time dimensions
reshaped_array = relevant_array.reshape(-1, relevant_array.shape[-1])
# Calculate the number of unique categories for each feature
num_categories = [len(np.unique(reshaped_array[:, i])) for i in range(reshaped_array.shape[1])]
print(num_categories)

# Set batch size
batch_size = 64
# Convert NumPy arrays to PyTorch tensors
train_X1 = train_X1.astype(float)
train_X2 = train_X2.astype(float)
input_tensor_train1 = torch.FloatTensor(train_X1)
input_tensor_train2 = torch.FloatTensor(train_X2)
target_tensor_train1 = torch.FloatTensor(train_Y1)
# Create a TensorDataset from input and target tensors
dataset_train = TensorDataset(input_tensor_train1, input_tensor_train2, target_tensor_train1)
# Create a DataLoader directly with input and target tensors
loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)

# Convert NumPy arrays to PyTorch tensors
val_X1 = val_X1.astype(float)
val_X2 = val_X2.astype(float)
input_tensor_val1 = torch.FloatTensor(val_X1)
input_tensor_val2 = torch.FloatTensor(val_X2)
target_tensor_val1 = torch.FloatTensor(val_Y1)
# Create a TensorDataset from input and target tensors
dataset_val = TensorDataset(input_tensor_val1, input_tensor_val2, target_tensor_val1)
# Create a DataLoader directly with input and target tensors
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True)

# Convert NumPy arrays to PyTorch tensors
test_X1 = test_X1.astype(float)
test_X2 = test_X2.astype(float)
input_tensor_test1 = torch.FloatTensor(test_X1)
input_tensor_test2 = torch.FloatTensor(test_X2)
target_tensor_test1 = torch.FloatTensor(test_Y1)
# Create a TensorDataset from input and target tensors
dataset_test = TensorDataset(input_tensor_test1, input_tensor_test2, target_tensor_test1)
# Create a DataLoader directly with input and target tensors
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

## Model initialization and loss function

class Autoencoder(nn.Module):
    def __init__(self, input_channels, image_height, image_width, num_numerical_features, latent_dim, num_categorical_features, embedding_dim):
        super(Autoencoder, self).__init__()
        # Embedding layers for tabular categorical data
        self.embedding_layers = nn.ModuleList([nn.Embedding(num_categories, embedding_dim) 
                                               for num_categories in num_categorical_features])
        
        # Encoder for image inputs
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Fully connected layer for tabular data
        self.fc_tabular = nn.Linear(embedding_dim * len(num_categorical_features) + num_numerical_features, 256)
        # Define the dimensions for the SelfAttention layer
        feature_dim = 32 * (image_height // 4) * (image_width // 3) + 256  # This should match the input dimension to fc_concat
        self.attention = SelfAttention(feature_dim)
        # print('feature_dim = ',feature_dim)
        
        # Concatenation layer
        self.fc_concat = nn.Linear(32 * (image_height // 4) * (image_width // 3) + 256, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * (image_height // 4) * (image_width // 3)),  # Adjust the output size accordingly
            nn.ReLU(),
            nn.Unflatten(1, (32, image_height // 4, image_width // 3)),
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 3), stride=(2, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(4, 3), stride=(2, 1), padding=(1, 0)),
            nn.Sigmoid()  # Use Sigmoid activation for pixel intensity range [0, 1]
        )

    def forward(self, x1, x2):
        # x1 is image input, x2 is tabular input
        # Split the first 24 features into two channels
        channel1 = x1[:, :, :6].unsqueeze(1)
        channel2 = x1[:, :, 6:12].unsqueeze(1)
        channel3 = x1[:, :, 12:18].unsqueeze(1)
        channel4 = x1[:, :, 18:24].unsqueeze(1)
        x_cnn = torch.cat((channel1, channel2, channel3, channel4), dim=1)       
        # Encode image
        encoded_image = self.encoder(x_cnn)
        # print(encoded_image.shape)
        # split numerical and categorical input from the tabular data
        numerical_input = x2[:,:3]
        categorical_input = x2[:, 3:]
        # Convert input to int type if necessary
        if categorical_input.dtype != torch.int32:
            categorical_input = categorical_input.to(torch.int32)
        # Embedding for categorical features
        embedded_categorical = [embedding_layer(categorical_input[:, i]) 
                                for i, embedding_layer in enumerate(self.embedding_layers)]
        # Concatenate embedded categorical data with continuous data
        tabular_input = torch.cat(embedded_categorical + [numerical_input], dim=-1)
        # Fully connected layer for tabular data
        tabular_out = self.fc_tabular(tabular_input)
        # Concatenate encoded image features with tabular data
        concatenated = torch.cat((encoded_image, tabular_out), dim=1)
        # print('concatenated shape = ',concatenated.shape)
        attention_out = self.attention(concatenated)
        latent = self.fc_concat(attention_out)
        # Decode
        decoded_image = self.decoder(latent)
        decoded_image = decoded_image.squeeze(1)
        return decoded_image

class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()

        self.query = nn.Linear(feature_dim, feature_dim // 8)
        self.key = nn.Linear(feature_dim, feature_dim // 8)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.softmax = nn.Softmax(dim=-1)  

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.size(-1))
        attention = self.softmax(attention_scores)
        attended_features = torch.matmul(attention, V)
        return attended_features
    
    
# custom_huber_loss
def custom_loss(y_pred, y_true, delta=0.1, positive_weight=0.1, negative_weight=1):
    # Mask for positions where y_true > 0
    mask = y_true > 0
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    residual = y_true_masked - y_pred_masked
    abs_residual = torch.abs(residual)
    quadratic_part = torch.clamp(abs_residual, max=delta)
    linear_part = abs_residual - quadratic_part
    
    # Apply different weights based on the sign of the error, positive means overpredict, negative means underpredict
    positive_mask = residual >= 0
    negative_mask = residual < 0

    # Calculate loss for positive errors, overprediction
    positive_loss = 0.5 * positive_weight * quadratic_part[positive_mask] ** 2 + delta * linear_part[positive_mask]
    # Calculate loss for negative errors, underprediction
    negative_loss = 0.5 * negative_weight * quadratic_part[negative_mask] ** 2 + delta * linear_part[negative_mask]
    # Concatenate positive and negative losses
    loss = torch.cat((positive_loss, negative_loss))
 
    return loss.mean()

## Model Training

# initialize the model and move it to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Instantiate the model
model = Autoencoder(input_channels=4, image_height = 48, image_width = 6, num_numerical_features = 3, num_categorical_features=[cate+1 for cate in num_categories], embedding_dim=2, latent_dim=128).to(device)

# Define your loss function (e.g., Mean Squared Error)
criterion = nn.MSELoss()

# Define your optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 120

tic = time.perf_counter()
finalloss_train = []
finalloss_val = []

# Define variables for tracking the best model and its performance
best_model_weights = None
best_val_loss = float('inf')
patience = 20  
counter = 0 
early_stop = False
for epoch in range(num_epochs):
    tic_ep = time.perf_counter()
    model.train()
    train_loss = 0.0  # initialize train_loss for each epoch
    for i, (inputs1, inputs2, targets1) in enumerate(loader_train):
        # move inputs and targets to the GPU
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        targets1 = targets1.to(device)
        # Forward pass
        outputs = model(inputs1, inputs2)
        # print(outputs.shape)
        loss = custom_loss(outputs, targets1)  # Calculate loss

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        train_loss += loss.item()  # accumulate loss for each batch
    # validate the model
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs1, inputs2, targets1 in loader_val:
            
            inputs1 = inputs1.to(device)
            inputs2 = inputs2.to(device)
            targets1 = targets1.to(device)
            outputs = model(inputs1, inputs2)
            
            loss = custom_loss(outputs, targets1)  # Calculate loss
            
            val_loss += loss.item()
    
    # toc_ep = time.perf_counter()
    # epoch_time = toc_ep-tic_ep
    
    # Check if the current model has the best validation performance
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = model.state_dict()
        counter = 0  
    else:
        counter += 1  
        
    # stop training if performance not improved after N=patience epochs
    if counter >= patience:
        print("Validation loss did not improve for {} epochs. Early stopping...".format(patience))
        early_stop = True
        break
    
    finalloss_train.append(train_loss/len(loader_train))
    finalloss_val.append(val_loss/len(loader_val))

    toc_ep = time.perf_counter()
    epoch_time = toc_ep-tic_ep
    
    if epoch % 5 == 0 or epoch == num_epochs-1:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(loader_train):.6f}, Val Loss: {val_loss/len(loader_val):.6f}, Time: {epoch_time:.2f} sec')
toc = time.perf_counter()
print(f"Triaining finished in {toc - tic:0.4f} seconds")
if not early_stop:
    print("Training finished without early stopping.")
torch.save(best_model_weights, 'trained_models/trained_model.pth')