import torch
import torch.nn as nn
from models import UNetGenerator, PatchGANDiscriminator  
from trainer import Trainer
from dataset import ColorizationDataset
from dataset import create_grayscale_dataset

# Define preprocessing transformations
bw_transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),   # Add this line for random horizontal flipping
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

color_transform = transforms.Compose([
    #transforms.RandomHorizontalFlip(),   # Add this line for random horizontal flipping
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])


create_grayscale_dataset(color_dir, grayscale_dir)

dataset_historical = ColorizationDataset(bw_dir, color_dir, use_modern_gray=False, bw_transform=bw_transform, color_transform=color_transform)
dataloader_historical = torch.utils.data.DataLoader(dataset_historical, batch_size=13, shuffle=True)
dataset_modern_gray = ColorizationDataset(grayscale_dir, color_dir, use_modern_gray=True, bw_transform=bw_transform, color_transform=color_transform)
dataloader_modern_gray = torch.utils.data.DataLoader(dataset_modern_gray, batch_size=13, shuffle=True)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create the generator and discriminator models
G = UNetGenerator(input_channels=1, output_channels=3).to(device)  # Assuming grayscale to RGB
D = PatchGANDiscriminator(input_channels=4).to(device)

# Define the optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Define the loss functions
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

# Create the Trainer instance with necessary parameters
trainer = Trainer(
    G, 
    D, 
    optimizer_G, 
    optimizer_D, 
    dataloader_historical=dataloader_historical, 
    dataloader_modern_gray=dataloader_modern_gray, 
    device=device, 
    lambda_L1=100, 
    criterion_GAN=criterion_GAN, 
    criterion_L1=criterion_L1, 
    log_wandb=True, 
    track_activations=True
)

# Train the model
trainer.train(start_epoch=0, num_epochs=100)
