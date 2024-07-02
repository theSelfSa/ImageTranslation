import torch
import pandas as pd
import os
from collections import defaultdict

# Function to load tensor from CSV file
def load_tensor_from_csv(file_path, shape):
    # Load the DataFrame from the CSV file
    df_loaded = pd.read_csv(file_path, header=None)
    
    # Convert the DataFrame to a NumPy array and reshape to original tensor shape
    tensor_np_loaded = df_loaded.values.reshape(shape)
    
    # Convert the NumPy array to a PyTorch tensor
    tensor = torch.tensor(tensor_np_loaded).float()  # Ensure the tensor is float
    
    # Move tensor to GPU if available
    if torch.cuda.is_available():
        tensor = tensor.to('cuda')
    
    return tensor

# Define the directory and file naming patterns
directory = "Exp/train"
file_patterns = {
    "pred_fake": "pred_fake_{}_{}.csv",
    "pred_real": "pred_real_{}_{}.csv",
    "fake_B": "fake_B_{}_{}.csv",
    "real_B": "real_B_{}_{}.csv"
}

# Extract unique image identifiers and run identifiers
files = os.listdir(directory)
identifiers = set((f.split('_')[2], f.split('_')[3].split('.')[0]) for f in files if f.startswith("pred_fake"))

# Define the loss functions
mse_loss = torch.nn.MSELoss()
l1_loss_fn = torch.nn.L1Loss()

# Data structures to store losses
D_losses = defaultdict(list)
G_losses = defaultdict(list)

# Prepare to save losses to CSV
losses_data = []
epoch = 1
# Loop through the unique identifiers and calculate losses
for image_id, run_id in identifiers:
    pred_fake_path = os.path.join(directory, file_patterns["pred_fake"].format(image_id, run_id))
    pred_real_path = os.path.join(directory, file_patterns["pred_real"].format(image_id, run_id))
    fake_B_path = os.path.join(directory, file_patterns["fake_B"].format(image_id, run_id))
    real_B_path = os.path.join(directory, file_patterns["real_B"].format(image_id, run_id))

    # Load tensors from CSV files
    pred_fake_loaded = load_tensor_from_csv(pred_fake_path, (1, 1, 30, 30))
    pred_real_loaded = load_tensor_from_csv(pred_real_path, (1, 1, 30, 30))
    fake_B_loaded = load_tensor_from_csv(fake_B_path, (3, 256, 256)).unsqueeze(0)
    real_B_loaded = load_tensor_from_csv(real_B_path, (3, 256, 256)).unsqueeze(0)

    # Create tensors of zeros and ones with the same shape
    zero_tensor = torch.zeros_like(pred_fake_loaded)
    one_tensor = torch.ones_like(pred_real_loaded)

    # Calculate MSE loss
    D_fake_mse_loss = mse_loss(pred_fake_loaded, zero_tensor)
    D_real_mse_loss = mse_loss(pred_real_loaded, one_tensor)
    G_fake_mse_loss = mse_loss(pred_fake_loaded, one_tensor)

    # Calculate L1 loss
    l1_loss_value = l1_loss_fn(real_B_loaded, fake_B_loaded)

    # Store the losses
    D_losses[image_id].append((D_fake_mse_loss.item(), D_real_mse_loss.item()))
    G_losses[image_id].append((l1_loss_value.item(),D_fake_mse_loss.item()))

    print(f"Image ID: {image_id}, Run ID: {run_id}")
    print(f"MSE Loss between pred_fake_loaded and zero_tensor: {D_fake_mse_loss.item()}")
    print(f"MSE Loss between pred_real_loaded and one_tensor: {D_real_mse_loss.item()}")
    print(f"L1 Loss between real_B_loaded and fake_B_loaded: {l1_loss_value.item()}")
    print(f"MSE Loss between pred_fake_loaded and one_tensor: {G_fake_mse_loss.item()}")

    losses_data.append([run_id, epoch, image_id, G_fake_mse_loss.item(), l1_loss_value.item()*100, (l1_loss_value.item() * 100)+ G_fake_mse_loss.item() , D_fake_mse_loss.item(), D_real_mse_loss.item(), 0.5*(D_fake_mse_loss.item() + D_real_mse_loss.item())])



# Create a DataFrame from the collected data
df_losses = pd.DataFrame(losses_data, columns=['run', 'epoch', 'image_id', 'G_GAN', 'G_L1', 'G', 'D_fake', 'D_real', 'D'])

# Save the DataFrame to CSV
csv_output_path = "Exp/train_losses_summary.csv"
df_losses.to_csv(csv_output_path, index=False)

print(f"Losses summary saved to {csv_output_path}")