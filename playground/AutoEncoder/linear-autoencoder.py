import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm
# Function to visualize the latent space
from umap import UMAP
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, latent_dim=32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()  # Output values between 0 and 1 for image reconstruction
        )
    
    def forward(self, x):
        # Encode
        z = self.encoder(x)
        # Decode
        reconstructed = self.decoder(z)
        return reconstructed, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)



# Function to load and preprocess data
def load_data(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, test_loader

# Training function
def train(model, train_loader, num_epochs=10, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on {device}")
    
    # Use a combination of MSE and L1 loss for better reconstruction
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()
    
    # Use Adam optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    # For tracking training progress
    loss_history = []
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for data, _ in pbar:
            # Reshape data and move to device
            img = data.view(data.size(0), -1).to(device)
            
            # Forward pass
            reconstructed, _ = model(img)
            
            # Combined loss
            mse_loss = mse_criterion(reconstructed, img)
            l1_loss = l1_criterion(reconstructed, img)
            loss = 0.8 * mse_loss + 0.2 * l1_loss
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # Update learning rate based on loss
        scheduler.step(avg_loss)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # Print statistics
        print(f'\nEpoch [{epoch+1}/{num_epochs}]')
        print(f'Average Loss: {avg_loss:.6f}')
        print(f'Best Loss: {best_loss:.6f}')
        print(f'Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        print(f'Epoch Time: {time.time() - epoch_start_time:.2f}s')
        print(f'Total Time: {(time.time() - start_time)/60:.2f}m')
        print('-' * 50)
    
    return model, loss_history

# Function to visualize original and reconstructed images
def visualize_reconstructions(model, test_loader, num_images=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get a batch from the test loader
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    
    # Select a subset of images to visualize
    images = images[:num_images]
    
    # Reshape and pass through the autoencoder
    with torch.no_grad():
        flattened_images = images.view(images.size(0), -1).to(device)
        reconstructed_images, _ = model(flattened_images)
        
        # Reshape back to original image dimensions
        reconstructed_images = reconstructed_images.view(images.size()).cpu()
    
    # Plot original vs reconstructed images
    plt.figure(figsize=(20, 4))
    
    for i in range(num_images):
        # Display original images
        ax = plt.subplot(2, num_images, i + 1)
        plt.imshow(images[i][0].numpy(), cmap='gray')
        plt.title('Original')
        plt.axis('off')
        
        # Display reconstructed images
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i][0].numpy(), cmap='gray')
        plt.title('Reconstructed')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()



def visualize_latent_space(model, test_loader, n_samples=2000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Lists to store latent representations and corresponding labels
    latent_representations = []
    labels = []
    
    # Process batches of test data
    with torch.no_grad():
        for data, label in test_loader:
            # Reshape and encode the data
            img = data.view(data.size(0), -1).to(device)
            z = model.encode(img).cpu().numpy()
            
            # Store the latent representations and labels
            latent_representations.append(z)
            labels.append(label.numpy())
            
            # Break if we have enough samples
            if len(np.concatenate(latent_representations)) >= n_samples:
                break
    
    # Concatenate the batches
    latent_representations = np.concatenate(latent_representations)[:n_samples]
    labels = np.concatenate(labels)[:n_samples]
    
    # Perform UMAP dimensionality reduction to two dimensions
    umap = UMAP(n_components=2)
    reduced_latent_representations = umap.fit_transform(latent_representations)
    
    # Create a scatter plot of the reduced latent space
    plt.figure(figsize=(10, 8))
    for i in range(10):  # Assuming 10 classes for digits 0-9
        indices = labels == i
        plt.scatter(reduced_latent_representations[indices, 0], reduced_latent_representations[indices, 1], 
                    label=str(i), alpha=0.5)
    plt.legend(title='Digit')
    plt.title('Latent Space Visualization with UMAP')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Main function to run the entire pipeline
def main():
    # Hyperparameters
    input_dim = 28 * 28  # MNIST image size
    hidden_dim = 128
    latent_dim = 32
    batch_size = 128
    num_epochs = 20
    learning_rate = 1e-3
    
    # Load data
    print("Loading data...")
    train_loader, test_loader = load_data(batch_size)
    
    # Initialize model
    model = Autoencoder(input_dim, hidden_dim, latent_dim)
    print(f"Model initialized with architecture:\nEncoder: {model.encoder}\nDecoder: {model.decoder}")
    
    # Train the model
    print("\nTraining the autoencoder...")
    model, loss_history = train(model, train_loader, num_epochs, learning_rate)
    
    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()
    
    # Visualize reconstructions
    print("\nVisualizing reconstructions...")
    visualize_reconstructions(model, test_loader)
    
    # Visualize latent space
    print("\nVisualizing latent space...")
    visualize_latent_space(model, test_loader)
    
    # Save the model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'autoencoder_model.pth')
    print("Model saved as 'autoencoder_model.pth'")

if __name__ == "__main__":
    main()
