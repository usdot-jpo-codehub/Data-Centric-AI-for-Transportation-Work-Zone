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

valtest_df_all = pd.read_csv('processed data/valtest_df_all.csv')
saved_best_model = torch.load('trained_models/trained_model.pth')

valtest_wzid = valtest_df_all['wz_id'].unique()
cong_valtest_wzid = valtest_df_all[(valtest_df_all['wz_id'].isin(impact_wzid))]['wz_id'].unique()

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

# initialize the model and move it to the GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Instantiate the model
model = Autoencoder(input_channels=4, image_height = 48, image_width = 6, num_numerical_features = 3, num_categorical_features=[cate+1 for cate in num_categories], embedding_dim=2, latent_dim=128).to(device)

model.load_state_dict(saved_best_model)



def pred_calculate(wzid,df_in):
    df = df_in[df_in['wz_id']==wzid].drop(['wz_id'], axis=1)
    input_array = df.values
    output_array = df[['speed_1','speed_2','speed_3','speed_4','speed_5','speed_6']].values
    window_length = 4
    sequences = []
    targets = []
    # Create sub-sequences and corresponding targets
    for j in range(len(input_array) - window_length):
        sub_sequence = input_array[:j + window_length]
        # Define the amount of padding for each dimension
        pad_amount = ((0, 52-sub_sequence.shape[0]), (0, 0))
        # Pad the array with zeros along the first dimension
        sub_sequence = np.pad(sub_sequence, pad_amount, mode='constant', constant_values=0)
        sequences.append(sub_sequence)
    # Convert sequences and targets to NumPy arrays
    targets = output_array[window_length:]
    sequences = np.array(sequences)
    example_X, example_Y = sequences.copy(), targets.copy()
    # Convert NumPy arrays to PyTorch tensors
    example_X = example_X.astype(float)
    input_tensor_example = torch.FloatTensor(example_X)
    target_tensor_example = torch.FloatTensor(example_Y)
    # Evaluate the model on the test set
    outputs_all = []   
    model.eval()
    for j in range(input_tensor_example.shape[0]):
        single_input = input_tensor_example[j].unsqueeze(0).clone()
        if j>0:
            single_input[0,window_length:window_length+j,:24] = outputs_all_tensor
        inputs = single_input.to(device)
        outputs = model(inputs)
        # Select the specified columns
        selected_columns = inputs[:, :, :6]
        # Compute the sum along the second dimension (columns)
        column_sum = torch.sum(selected_columns, dim=1)
        # Create a mask indicating zero columns (0 if zero column, 1 otherwise)
        mask = torch.where(column_sum == 0, torch.tensor(0), torch.tensor(1))
        mask_output = outputs[0]*mask
        # Define a small tolerance threshold
        tolerance = 1e-6
        # Round values close to zero to zero
        mask_output[mask_output.abs() < tolerance] = 0
        predict_spd = inputs[0,0,:24]
        # Define the indices to replace in tensor A
        # indices_to_replace = [0, 4, 8, 12, 16, 20]
        indices_to_replace = [0, 1, 2, 3, 4, 5]
        # Replace the values in tensor A with the values from tensor B
        predict_spd[indices_to_replace] = mask_output.squeeze()
        predict_spd = predict_spd.tolist()
        outputs_all.append(predict_spd) 
        outputs_all_tensor = torch.tensor(list(outputs_all))

    targets = target_tensor_example.squeeze().tolist()
    target_encoded = targets.copy()
    targets = [[x * spd_diff + spd_min for x in row] for row in targets]
    # outputs_speed = [[row[i] for i in [0, 4, 8, 12, 16, 20]] for row in outputs_all]
    outputs_speed = [[row[i] for i in [0, 1, 2, 3, 4, 5]] for row in outputs_all]
    output_regressive = [[x * spd_diff + spd_min for x in row] for row in outputs_speed]
    return output_regressive,targets, df, outputs_speed,target_encoded


## Define the name of the new folder
folder_name = 'test_plots'
# Define the root path where you want to create the new folder
root_path = "plots"
# Concatenate the root path with the folder name
new_folder_path = os.path.join(root_path, folder_name)
# Check if the folder already exists, if not, create it
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)
    print(f"Folder '{folder_name}' created successfully under '{root_path}'.")
else:
    print(f"Folder '{folder_name}' already exists under '{root_path}'.")

# test on impacted WZs
for i in range(len(cong_valtest_wzid)):
    wzid = cong_valtest_wzid[i]
    regression_output, target_output, raw_df, _,_ = pred_calculate(wzid,valtest_df_all)

    speed_data_raw = ((raw_df[['speed_1','speed_2','speed_3','speed_4','speed_5','speed_6']]*spd_diff+spd_min).values - 
                   (raw_df[['avgspd_1','avgspd_2','avgspd_3','avgspd_4','avgspd_5','avgspd_6']]*spd_diff+spd_min).values)
    speed_data1 = speed_data_raw.copy()
    speed_data1 = speed_data1[(len(speed_data1)-len(regression_output)):]
    speed_data2 = (np.array(regression_output) -
                  (raw_df[['avgspd_1','avgspd_2','avgspd_3','avgspd_4','avgspd_5','avgspd_6']]*spd_diff+spd_min).values[(len(speed_data_raw)-len(regression_output)):])
    plot_heatmap_savelocal(speed_data1,speed_data2, wzid,'plots/'+folder_name)