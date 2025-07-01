# Getting unique samples for SHAP analysis top 6 labels 
import random
import shap
import pandas as pd
from src.utils import convert_tensor
import matplotlib.pyplot as plt
import numpy as np

mlb = pd.read_pickle('./data/mlb.pkl')
idx2label = pd.read_pickle('./data/idx2label.pkl')

# Getting unique samples for SHAP analysis 
test_data = pd.read_pickle('./savepoints/0/test_data_fold_0.pkl')
x_test = test_data[0] 
y_test = test_data[1] 

# Check the shape of the slices
print(f"x_test shape: {x_test.shape}")  
print(f"y_test shape: {y_test.shape}")  

# Transform the labels to the binary matrix representation
y_transformed = mlb.transform(idx2label[y_test])
print("Shape of y_transformed:", y_transformed.shape)

# Define the labels of interest
labels_of_interest = [73, 68, 100, 43, 104, 99]

# Set the random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# Define number of samples you want total
total_samples = 50
num_labels = len(labels_of_interest)
samples_per_label = total_samples // num_labels

# List to store the selected indices
test_indices = []

for label in labels_of_interest:
    label_idx = label
    
    # Get the indices where this label is present
    label_indices = np.where(y_transformed[:, label_idx] == 1)[0]
    
    # If there are enough, randomly choose samples_per_label
    if len(label_indices) >= samples_per_label:
        selected = np.random.choice(label_indices, samples_per_label, replace=False)
    else:
        # If not enough, take all available
        selected = label_indices
    
    test_indices.extend(selected)

# Shuffle in case of any overlap/repeats across labels
test_indices = np.unique(test_indices)
np.random.shuffle(test_indices)

# If we ended up with more than 50 (due to overlap), trim down to exactly 50
if len(test_indices) > total_samples:
    test_indices = test_indices[:total_samples]

print(f"Selected {len(test_indices)} test samples for SHAP.")

# Select features and corresponding labels based on selected indices
x_shap = x_test[test_indices]  # Select the features
y_shap = y_test[test_indices]  # Select the corresponding labels

print(f"x_shap shape: {x_shap.shape}")
print(f"y_shap shape: {y_shap.shape}")

# Save the selected data for SHAP analysis
np.savez("shap_test_final.npz", x_shap=x_shap, y_shap=y_shap)

# Getting unique background samples for SHAP analysis top 6
train_data = pd.read_pickle('./savepoints/0/train_data_fold_0.pkl')
x_train = train_data[0]  
y_train = train_data[1]

# Check the shape of the slices
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")

# Transform the labels to the binary matrix representation (if necessary)
y_transformed_train = mlb.transform(idx2label[y_train])
print("Shape of y_transformed_train:", y_transformed_train.shape)

# Define the labels of interest
labels_of_interest = [73, 68, 100, 43, 104, 99]

# Set the random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)

# Define number of samples you want total
total_samples = 50
num_labels = len(labels_of_interest)
samples_per_label = total_samples // num_labels

# List to store the selected indices
background_indices = []

for label in labels_of_interest:
    label_idx = label
    
    # Get the indices where this label is present
    label_indices = np.where(y_transformed_train[:, label_idx] == 1)[0]
    
    # If there are enough, randomly choose samples_per_label
    if len(label_indices) >= samples_per_label:
        selected = np.random.choice(label_indices, samples_per_label, replace=False)
    else:
        # If not enough, take all available
        selected = label_indices
    
    background_indices.extend(selected)

# Shuffle in case of any overlap/repeats across labels
background_indices = np.unique(background_indices)
np.random.shuffle(background_indices)

# If we ended up with more than 50 (due to overlap), trim down to exactly 50
if len(background_indices) > total_samples:
    background_indices = background_indices[:total_samples]

print(f"Selected {len(background_indices)} background samples for SHAP.")

# Select features and corresponding labels based on selected indices
x_background_shap = x_train[background_indices]  # Select the features
y_background_shap = y_train[background_indices]  # Select the corresponding labels

print(f"x_background_shap shape: {x_background_shap.shape}")
print(f"y_background_shap shape: {y_background_shap.shape}")

# Save the selected background data for SHAP analysis
np.savez("shap_train_final.npz", x_background=x_background_shap, y_background=y_background_shap)
