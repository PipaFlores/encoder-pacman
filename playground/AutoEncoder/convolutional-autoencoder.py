import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import time
from umap import UMAP
from tqdm import tqdm
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the Convolutional Autoencoder architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, dropout_rate = 0.1):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # [batch, 16, 28, 28]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # [batch, 16, 14, 14]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # [batch, 32, 14, 14]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # [batch, 32, 7, 7]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), # [batch, 64, 7, 7]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1), # [batch, 32, 7, 7]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),                    # [batch, 32, 14, 14]
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1), # [batch, 16, 14, 14]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),                    # [batch, 16, 28, 28]
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),  # [batch, 1, 28, 28]
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

                # Add a bottleneck layer
        self.bottleneck = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 7 * 7),
            nn.Unflatten(1, (64, 7, 7))
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        bottleneck = self.bottleneck(encoded)
        decoded = self.decoder(bottleneck)
        return decoded, bottleneck
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def get_latent(self, x):
        encoded = self.encoder(x)
        return self.bottleneck(encoded) 

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
            # Move data to device
            img = data.to(device)
            
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
    images = images[:num_images].to(device)
    
    # Pass through the autoencoder
    with torch.no_grad():
        reconstructed_images, _ = model(images)
        reconstructed_images = reconstructed_images.cpu()
        images = images.cpu()
    
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

# Function to visualize feature maps
def visualize_feature_maps(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    # Get a single image
    dataiter = iter(test_loader)
    images, _ = next(dataiter)
    img = images[0:1].to(device)  # Use just the first image
    
    # Get the feature maps from the encoder
    with torch.no_grad():
        feature_maps, _ = model(img)
        
        # Get the intermediate feature maps
        # We'll need to make a new forward hook to get these
        feature_maps = []
        
        def hook_fn(module, input, output):
            feature_maps.append(output)
        
        # Register hooks for each conv layer in the encoder
        hooks = []
        for layer in model.encoder:
            if isinstance(layer, nn.Conv2d):
                hooks.append(layer.register_forward_hook(hook_fn))
        
        # Forward pass to get feature maps
        model(img)
        
        # Remove the hooks
        for hook in hooks:
            hook.remove()
    
    # Visualize the feature maps from the first convolutional layer
    if feature_maps:
        first_layer_features = feature_maps[0][0].cpu().numpy()
        
        # Create a figure to display the feature maps
        fig = plt.figure(figsize=(15, 10))
        
        # Plot the original image
        plt.subplot(4, 5, 1)
        plt.imshow(img[0, 0].cpu().numpy(), cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Plot the feature maps
        num_features = min(16, first_layer_features.shape[0])
        for i in range(num_features):
            plt.subplot(4, 5, i + 2)
            plt.imshow(first_layer_features[i], cmap='viridis')
            plt.title(f'Feature {i+1}')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Function to explore the latent space
def visualize_latent_space(model, test_loader, n_samples=1000):
    from sklearn.manifold import TSNE
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    latent_representations = []
    labels = []
    
    with torch.no_grad():
        for data, label in test_loader:
            img = data.to(device)
            
            # Get bottleneck representation
            latent = model.get_latent(img)
            
            # Flatten if necessary (should already be flat from the bottleneck)
            flattened = latent.reshape(latent.size(0), -1).cpu().numpy()
            
            latent_representations.append(flattened)
            labels.append(label.numpy())
            
            if len(latent_representations) * data.shape[0] >= n_samples:
                break
    
    # Rest of the function remains the same
    latent_representations = np.concatenate(latent_representations)[:n_samples]
    labels = np.concatenate(labels)[:n_samples]
    print("Applying UMAP to visualize high-dimensional latent space...")
    umap = UMAP(n_components=2, random_state=42)
    latent_umap = umap.fit_transform(latent_representations)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(latent_umap[:, 0], latent_umap[:, 1], 
                         c=labels, cmap='viridis', alpha=0.5)
    plt.colorbar(scatter, label='Digit')
    plt.title('UMAP Visualization of Latent Space')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    

# Main function to run the entire pipeline
def main():
    # Hyperparameters
    batch_size = 128
    num_epochs = 15
    learning_rate = 1e-3
    
    # Load data
    print("Loading data...")
    train_loader, test_loader = load_data(batch_size)
    
    # Initialize model
    model = ConvAutoencoder()
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
    
    # Visualize feature maps
    print("\nVisualizing feature maps...")
    visualize_feature_maps(model, test_loader)
    
    # Visualize latent space
    print("\nVisualizing latent space using t-SNE...")
    visualize_latent_space(model, train_loader)
    
    # Save the model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'conv_autoencoder_model.pth')
    print("Model saved as 'conv_autoencoder_model.pth'")

if __name__ == "__main__":
    main()
