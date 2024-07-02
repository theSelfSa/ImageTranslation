import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data into a DataFrame
file_path = 'data.csv'  # Replace with your CSV file path
data = pd.read_csv(file_path)

# Plot and save the training and validation loss for the Discriminator
plt.figure(figsize=(8, 6))
plt.plot(data['epoch'], data['D_train'], label='D_train', marker='o')
plt.plot(data['epoch'], data['D'], label='D', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Discriminator Loss')
plt.legend()
plt.savefig('discriminator_loss.png')
plt.close()

# Plot and save the training and validation loss for the Generator
plt.figure(figsize=(8, 6))
plt.plot(data['epoch'], data['G_train'], label='G_train', marker='o')
plt.plot(data['epoch'], data['G'], label='G', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator Loss')
plt.legend()
plt.savefig('generator_loss.png')
plt.close()

