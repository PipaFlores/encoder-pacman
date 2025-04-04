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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set random seed for reproducibility
torch.manual_seed(42)


class TransformerAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim=784,
        hidden_dim=256,
        latent_dim=32,
        nhead=8,
        num_layers=3,
        dropout=0.1,
    ):
        super(TransformerAutoencoder, self).__init__()

        # MNIST specific: reshape 28x28 into 7x7 patches of size 4x4 (16 values per patch)
        self.patch_size = 4
        self.num_patches = 49  # 7x7 grid of patches
        self.hidden_dim = hidden_dim

        # Learnable positional encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, hidden_dim))

        # Patch embedding
        self.patch_embedding = nn.Sequential(
            nn.Linear(16, hidden_dim),  # 16 = 4x4 patch size
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Simplified encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,  # Reduced from 4x to 2x
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Simpler latent projection
        self.latent_projection = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim), nn.LayerNorm(latent_dim)
        )

        # Simpler decoder
        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Simplified output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, 16),  # Direct projection to patch size
            nn.Sigmoid(),
        )

    def _to_patches(self, x):
        # x shape: (batch_size, 784)
        batch_size = x.size(0)
        # Reshape to (batch_size, 28, 28)
        x = x.view(batch_size, 28, 28)
        # Extract 4x4 patches
        patches = x.unfold(1, self.patch_size, self.patch_size).unfold(
            2, self.patch_size, self.patch_size
        )
        # Reshape to (batch_size, 49, 16)
        patches = patches.contiguous().view(
            batch_size, -1, self.patch_size * self.patch_size
        )
        return patches

    def _from_patches(self, patches, batch_size):
        # patches shape: (batch_size, 49, 16)
        # Reshape to (batch_size, 7, 7, 4, 4)
        x = patches.view(batch_size, 7, 7, 4, 4)
        # Combine patches back to image
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.view(batch_size, 28, 28)
        # Flatten to (batch_size, 784)
        return x.view(batch_size, -1)

    def forward(self, x):
        batch_size = x.size(0)

        # Convert input to patches
        patches = self._to_patches(x)

        # Embed patches
        x = self.patch_embedding(patches)

        # Add positional embeddings
        x = x + self.pos_embedding

        # Encode
        encoded = self.encoder(x)

        # Get latent representation from CLS token (first token)
        z = encoded.mean(dim=1)  # Use mean pooling
        z = self.latent_projection(z)

        # Decode
        decoder_input = (
            self.latent_to_hidden(z).unsqueeze(1).repeat(1, self.num_patches, 1)
        )
        decoded = self.decoder(decoder_input, encoded)
        patches_out = self.output_projection(decoded)

        # Convert patches back to image
        reconstructed = self._from_patches(patches_out, batch_size)

        return reconstructed, z

    def encode(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1, 16)
        x = self.input_projection(x)
        x = x + self.pos_encoder
        encoded = self.encoder(x)
        z = encoded.mean(dim=1)
        return self.latent_projection(z)

    def decode(self, z):
        batch_size = z.size(0)
        hidden = self.latent_to_hidden(z)
        decoder_input = hidden.unsqueeze(1).repeat(1, 49, 1)  # 49 patches
        decoded = self.decoder(decoder_input, decoder_input)
        decoded = self.output_projection(decoded)
        return decoded.reshape(batch_size, -1)


# Function to load and preprocess data
def load_data(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # MNIST dataset
    train_dataset = datasets.MNIST(
        root="./data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# Training function
def train(model, train_loader, num_epochs=10, learning_rate=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Training on {device}")

    # Use a combination of MSE and L1 loss for better reconstruction
    mse_criterion = nn.MSELoss()
    l1_criterion = nn.L1Loss()

    # Use AdamW with weight decay for better regularization
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # For tracking training progress
    loss_history = []
    best_loss = float("inf")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
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

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)

        # Update learning rate based on loss
        scheduler.step(avg_loss)

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "best_model.pth")

        # Print statistics
        print(f"\nEpoch [{epoch + 1}/{num_epochs}]")
        print(f"Average Loss: {avg_loss:.6f}")
        print(f"Best Loss: {best_loss:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Epoch Time: {time.time() - epoch_start_time:.2f}s")
        print(f"Total Time: {(time.time() - start_time) / 60:.2f}m")
        print("-" * 50)

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
        plt.imshow(images[i][0].numpy(), cmap="gray")
        plt.title("Original")
        plt.axis("off")

        # Display reconstructed images
        ax = plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_images[i][0].numpy(), cmap="gray")
        plt.title("Reconstructed")
        plt.axis("off")

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
    scatter = plt.scatter(
        reduced_latent_representations[:, 0],
        reduced_latent_representations[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.5,
    )
    plt.colorbar(scatter, label="Digit")
    plt.title("Latent Space Visualization with UMAP")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Main function to run the entire pipeline
def main():
    # Hyperparameters
    input_dim = 28 * 28  # MNIST image size
    hidden_dim = 256  # Reduced from 512
    latent_dim = 32  # Reduced from 64
    batch_size = 128
    num_epochs = 2  # Increased from 2
    learning_rate = 2e-4  # Adjusted learning rate

    # Load data
    print("Loading data...")
    train_loader, test_loader = load_data(batch_size)

    # Initialize model
    model = TransformerAutoencoder(input_dim, hidden_dim, latent_dim)
    print(
        f"Model initialized with architecture:\nEncoder: {model.encoder}\nDecoder: {model.decoder}"
    )

    # Train the model
    print("\nTraining the autoencoder...")
    model, loss_history = train(model, train_loader, num_epochs, learning_rate)

    # Plot the training loss
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
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
    torch.save(model.state_dict(), "autoencoder_model.pth")
    print("Model saved as 'autoencoder_model.pth'")


if __name__ == "__main__":
    main()
