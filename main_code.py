import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import numpy as np
import os
import time

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from torch.autograd import Variable
from PIL import Image
import fastcore.all as fc
import torch.nn.init as init

import fastprogress
from fastprogress import master_bar, progress_bar
import wandb
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math


device = torch.device("cuda:0" if torch.cuda.is_available() else "mps")
device

class ColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, bw_dir, color_dir, bw_transform=None, color_transform=None):
        self.bw_dir = bw_dir
        self.color_dir = color_dir
        self.image_files = [f for f in os.listdir(bw_dir) if f.endswith('.jpg')]
        self.bw_transform = bw_transform
        self.color_transform = color_transform

        # verify image path
        for img_file in self.image_files:
            bw_path = os.path.join(bw_dir, img_file)
            color_path = os.path.join(color_dir, img_file.replace('.jpg', '_result.jpg'))
            assert os.path.exists(bw_path) and os.path.exists(color_path), f"Image pair not found for {img_file}"

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        bw_path = os.path.join(self.bw_dir, self.image_files[idx])
        color_path = os.path.join(self.color_dir, self.image_files[idx].replace('.jpg', '_result.jpg'))

        # error handling
        try:
            bw_image = Image.open(bw_path).convert("L")
            color_image = Image.open(color_path).convert("RGB")
        except:
            print(f'Error Loading images at {bw_path} or {color_path}')
            return None, None
        
        seed = torch.randint(0, 2**32, ())
        torch.manual_seed(seed)
        if self.bw_transform:
            bw_image = self.bw_transform(bw_image)

        torch.manual_seed(seed)
        if self.color_transform:
            color_image = self.color_transform(color_image)

        return bw_image, color_image
    
    def set_transform(self, bw_transform, color_transform):
        self.bw_transform = bw_transform
        self.color_transform = color_transform


# Define preprocessing transformations
bw_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

color_transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
])

# Create dataset & dataloader instances
bw_dir = "/Users/parkermoesta/pix2pix/bw_images"
color_dir = "/Users/parkermoesta/pix2pix/color_images"
dataset = ColorizationDataset(bw_dir, color_dir, bw_transform=bw_transform, color_transform=color_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)



class Visualizer:
    def __init__(self, unnormalize=True):
        self.unnormalize = unnormalize

    def _unnormalize(self, img):
        if self.unnormalize:
            img = img * 0.5 + 0.5
        return img.clamp(0, 1)
    
    def display_images(self, bw_images, color_images, nrows=3):
        bw_images = self._unnormalize(bw_images)
        color_images = self._unnormalize(color_images)

        grid_bw = torchvision.utils.make_grid(bw_images[:nrows], nrows=nrows).permute(1,2,0).detach().cpu().numpy()
        grid_color = torchvision.utils.make_grid(color_images[:nrows], nrows=nrows).permute(1,2,0).detach().cpu().numpy()

        fig, ax = plt.subplots(2, 1, figsize=(12,8))

        ax[0].imshow(grid_bw, cmap='gray')
        ax[0].set_title('Black and White Images')
        ax[0].axis('off')

        ax[1].imshow(grid_color)
        ax[1].set_title('Color Images')
        ax[1].axis('off')

        plt.show()

    def display_generated(self, bw_images, color_images, generated_images, nrows=3):
        bw_images = self._unnormalize(bw_images)
        color_images = self._unnormalize(color_images)
        generated_images = self._unnormalize(generated_images)

        grid_bw = torchvision.utils.make_grid(bw_images[:nrows], nrow=nrows).permute(1,2,0).detach().cpu().numpy()
        grid_color = torchvision.utils.make_grid(color_images[:nrows], nrow=nrows).permute(1,2,0).detach().cpu().numpy()
        grid_gen = torchvision.utils.make_grid(generated_images[:nrows], nrow=nrows).permute(1,2,0).detach().cpu().numpy()

        fig, ax = plt.subplots(3, 1, figsize=(12,12))

        ax[0].imshow(grid_bw, cmap='gray')
        ax[0].set_title('Black and White Images')
        ax[0].axis('off')

        ax[1].imshow(grid_color)
        ax[1].set_title('Ground Truth Color Images')
        ax[1].axis('off')

        ax[2].imshow(grid_gen)
        ax[2].set_title('Generated Color Images')
        ax[2].axis('off')

        plt.show()

    def log_images_to_wandb(self, bw_images, color_images, generated_images, step):
        # Convert images to the format wandb expects (CHW -> HWC)
        bw_images = [img.permute(1,2,0) for img in bw_images]
        color_images = [img.permute(1,2,0) for img in color_images]
        generated_images = [img.permute(1,2,0) for img in generated_images]
        
        # Log images to wandb
        wandb.log({
            "Black and White": [wandb.Image(img, caption="Black and White") for img in bw_images],
            "Ground Truth Color": [wandb.Image(img, caption="Ground Truth Color") for img in color_images],
            "Generated Color": [wandb.Image(img, caption="Generated Color") for img in generated_images]
        }, step=step)


viz = Visualizer()
bw_images, color_images = next(iter(dataloader))  # get a batch of images
viz.display_images(bw_images, color_images)



def show_image(im, ax=None, figsize=None, title=None, noframe=True, **kwargs):
    "Show a PIL or PyTorch image on `ax`."
    if fc.hasattrs(im, ('cpu','permute','detach')):
        im = im.detach().cpu()
        if len(im.shape)==3 and im.shape[0]<5: im=im.permute(1,2,0)
    elif not isinstance(im,np.ndarray): im=np.array(im)
    if im.shape[-1]==1: im=im[...,0]
    if ax is None: _,ax = plt.subplots(figsize=figsize)
    ax.imshow(im, **kwargs)
    if title is not None: ax.set_title(title)
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    if noframe: ax.axis('off')
    return ax

def subplots(
    nrows:int=1, # Number of rows in returned axes grid
    ncols:int=1, # Number of columns in returned axes grid
    figsize:tuple=None, # Width, height in inches of the returned figure
    imsize:int=3, # Size (in inches) of images that will be displayed in the returned figure
    suptitle:str=None, # Title to be set to returned figure
    **kwargs
): # fig and axs
    "A figure and set of subplots to display images of `imsize` inches"
    if figsize is None: figsize=(ncols*imsize, nrows*imsize)
    fig,ax = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    if suptitle is not None: fig.suptitle(suptitle)
    if nrows*ncols==1: ax = np.array([ax])
    return fig,ax

def get_grid(
    n:int, # Number of axes
    nrows:int=None, # Number of rows, defaulting to `int(math.sqrt(n))`
    ncols:int=None, # Number of columns, defaulting to `ceil(n/rows)`
    title:str=None, # If passed, title set to the figure
    weight:str='bold', # Title font weight
    size:int=14, # Title font size
    **kwargs,
): # fig and axs
    "Return a grid of `n` axes, `rows` by `cols`"
    if nrows: ncols = ncols or int(np.floor(n/nrows))
    elif ncols: nrows = nrows or int(np.ceil(n/ncols))
    else:
        nrows = int(math.sqrt(n))
        ncols = int(np.floor(n/nrows))
    fig,axs = subplots(nrows, ncols, **kwargs)
    for i in range(n, nrows*ncols): axs.flat[i].set_axis_off()
    if title is not None: fig.suptitle(title, weight=weight, size=size)
    return fig,axs

def get_hist(h): return torch.stack(h.stats[2]).t().float().log1p()

def get_min(h):
    h1 = torch.stack(h.stats[2]).t().float()
    return h1[0]/h1.sum(0)


class Callback:
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class ModelCheckpoint(Callback):
    def __init__(self, filepath, monitor='loss_G', save_best_only=False):
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.best = float('inf')

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if not self.save_best_only or current < self.best:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': self.trainer.G.state_dict(),
                'discriminator_state_dict': self.trainer.D.state_dict(),
                'optimizer_G_state_dict': self.trainer.optimizer_G.state_dict(),
                'optimizer_D_state_dict': self.trainer.optimizer_D.state_dict(),
                'loss_D': logs['loss_D'],
                'loss_G': logs['loss_G']
                }, self.filepath.format(epoch=epoch, **logs))
            self.best = current

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop
    
class HooksCallback:
    def __init__(self, hook_fn, modules):
        self.hooks = []
        for module in modules:
            self.hooks.append(module.register_forward_hook(hook_fn))

    def remove(self):
        for hook in self.hooks:
            hook.remove()

def append_stats(module, inp, outp):
    if not hasattr(module, 'stats'): module.stats = ([], [])
    acts = outp.cpu()
    module.stats[0].append(acts.mean().item())
    module.stats[1].append(acts.std().item())

class ActivationStats(HooksCallback):
    def __init__(self, modules):
        super().__init__(append_stats, modules)
        self.modules = modules 
        self.hooks = [append_stats for _ in modules]

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def color_dim(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flat, self):
            show_image(get_hist(h), ax, origin='lower')

    def dead_chart(self, figsize=(11,5)):
        fig,axes = get_grid(len(self), figsize=figsize)
        for ax,h in zip(axes.flatten(), self):
            ax.plot(get_min(h))
            ax.set_ylim(0,1)


    def plot_stats(self, figsize=(10,4)):
        fig,axs = plt.subplots(1,2, figsize=figsize)
        for hook in self.hooks:
            for i in [0, 1]: 
                axs[i].plot(hook.stats[i])
        axs[0].set_title('Means')
        axs[1].set_title('Stdevs')
        plt.legend(range(len(self.hooks)))


class Trainer:
    def __init__(self, G, D, optimizer_G, optimizer_D, dataloader, device, lambda_L1,criterion_GAN, criterion_L1,log_wandb=True, track_activations=False, output_path="."):
        self.G = G
        self.D = D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        self.dataloader = dataloader
        self.device = device
        self.lambda_L1 = lambda_L1
        self.log_wandb = log_wandb
        self.criterion_GAN = criterion_GAN
        self.criterion_L1 = criterion_L1
        self.act_stats = ActivationStats([layer for layer in self.G.children() if isinstance(layer, nn.ReLU)])
        self.output_path = output_path
        self.callbacks = []
        self.G = G.to(device)
        self.D = D.to(device)
        # lr schedulers
        self.scheduler_G = ReduceLROnPlateau(self.optimizer_G, mode='min', factor=0.1, patience=10, verbose=True)
        self.scheduler_D = ReduceLROnPlateau(self.optimizer_D, mode='min', factor=0.1, patience=10, verbose=True)


        self.act_stats_gen = ActivationStats(list(self.G.children()))
        self.act_stats_disc = ActivationStats(list(self.D.children()))

        # early stopping
        self.early_stopping = EarlyStopping(patience=100, verbose=True)
        
        if log_wandb:
            wandb.watch(G, log="all")
            wandb.watch(D, log="all")

        if track_activations:
            modules_to_track = list(self.G.children()) + list(self.D.children())  # Change this if you want to track specific modules
            self.activation_stats = ActivationStats(modules_to_track)
            self.add_callback(self.activation_stats)

    def plot_activation_stats(self):
        self.activation_stats.plot_stats()

    def plot_color_dim(self):
        self.activation_stats.color_dim()

    def plot_dead_activations(self):
        self.activation_stats.dead_chart()

    def add_callback(self, callback):
        callback.trainer = self
        self.callbacks.append(callback)

    def luminance_loss(self, fake_color, bw_image):
        # Convert the generated color image to grayscale
        fake_gray = torch.mean(fake_color, dim=1, keepdim=True)
        return self.criterion_L1(fake_gray, bw_image)
    
    def weighted_L1_loss(self, fake_color, real_color, bw_image, alpha=0.5):
        # Standard L1 loss
        l1 = self.criterion_L1(fake_color, real_color)
        
        # Luminance loss
        lum_loss = self.luminance_loss(fake_color, bw_image)
        
        return l1 + alpha * lum_loss

    def train_step(self, epoch):
        epoch_loss_G = 0
        epoch_loss_D = 0

        for batch_idx, (bw_images, color_images) in enumerate(self.dataloader):
            # Set model inputs
            bw_images, color_images = bw_images.to(self.device), color_images.to(self.device)

            # generator forward pass
            fake_images = self.G(bw_images)

            if batch_idx % 200 == 0:  # Adjust this number to control the logging frequency
                #unique_step = len(self.dataloader) * epoch + batch_idx  # Compute a unique step for every epoch and batch combination
                self.log_images(bw_images, color_images, fake_images)

            # Discriminator training
            self.D.zero_grad()
            real_preds = self.D(torch.cat([bw_images, color_images], dim=1))
            fake_preds = self.D(torch.cat([bw_images, fake_images.detach()], dim=1))

            # calculating loss
            loss_D_real = self.criterion_GAN(real_preds, torch.ones_like(real_preds).to(self.device))
            loss_D_fake = self.criterion_GAN(fake_preds, torch.zeros_like(fake_preds).to(self.device))
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            epoch_loss_D += loss_D.item()

            loss_D.backward()
            self.optimizer_D.step()

            # generator training
            self.G.zero_grad()
            outputs_fake = self.D(torch.cat([bw_images, fake_images], dim=1))
            loss_GAN = self.criterion_GAN(outputs_fake, torch.ones_like(outputs_fake).to(self.device))
            loss_L1 = self.weighted_L1_loss(fake_images, color_images, bw_images, alpha=0.86)
            loss_G = loss_GAN + self.lambda_L1 * loss_L1
            epoch_loss_G += loss_G.item()

            loss_G.backward()
            self.optimizer_G.step()

        self.log_activation_stats()

        return {"loss_G": epoch_loss_G / len(self.dataloader), "loss_D": epoch_loss_D / len(self.dataloader)}


    def log_metrics(self, metrics):
        if self.log_wandb:
            wandb.log(metrics)

    def log_images(self, bw_images, color_images, fake_images):
        if self.log_wandb:
            # Convert tensors to numpy arrays and transpose the axes to (H, W, C)
            bw_images_list = [wandb.Image(img.transpose(1, 2, 0)) for img in bw_images.detach().cpu().numpy()]
            color_images_list = [wandb.Image(img.transpose(1, 2, 0)) for img in color_images.detach().cpu().numpy()]
            fake_images_list = [wandb.Image(img.transpose(1, 2, 0)) for img in fake_images.detach().cpu().numpy()]
            
            # Log images to WandB
            wandb.log({
                "BW Images": bw_images_list, 
                "Real Color Images": color_images_list, 
                "Generated Color Images": fake_images_list
            })


                
    def log_activation_stats(self):
        if self.log_wandb:
            log_data = {}

            # Logging for the generator
            for idx, module in enumerate(self.act_stats_gen.modules):
                mean = np.mean(module.stats[0])
                std = np.mean(module.stats[1])
                log_data[f"Mean_Gen_Layer_{idx}"] = mean
                log_data[f"Std_Gen_Layer_{idx}"] = std

            # Logging for the discriminator
            for idx, module in enumerate(self.act_stats_disc.modules):
                mean = np.mean(module.stats[0])
                std = np.mean(module.stats[1])
                log_data[f"Mean_Disc_Layer_{idx}"] = mean
                log_data[f"Std_Disc_Layer_{idx}"] = std

            # Log the data
            wandb.log(log_data)
    def save_checkpoint(self, epoch, metrics):
        checkpoint_path = f'checkpoint_{epoch}.pth'
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.G.state_dict(),
            'discriminator_state_dict': self.D.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'loss_D': metrics['loss_D'],
            'loss_G': metrics['loss_G'],
        }, checkpoint_path)
        if self.log_wandb:
            wandb.save(checkpoint_path)

    def find_lr(self, init_value=1e-8, final_value=10.0, beta=0.98):
        num = len(self.dataloader) - 1
        mult = (final_value / init_value) ** (1/num)
        lr = init_value
        self.optimizer_G.param_groups[0]['lr'] = lr
        avg_loss = 0.0
        best_loss = 0.0
        batch_num = 0
        losses = []
        log_lrs = []
        for data in self.dataloader:
            batch_num += 1
            # Get the inputs to the device
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            # Forward pass
            self.optimizer_G.zero_grad()
            outputs = self.G(inputs)
            loss = self.criterion_GAN(outputs, labels)
            
            # Compute the smoothed loss
            avg_loss = beta * avg_loss + (1-beta) *loss.item()
            smoothed_loss = avg_loss / (1 - beta**batch_num)
            
            # Stop if the loss is exploding
            if batch_num > 1 and smoothed_loss > 4 * best_loss:
                return log_lrs, losses
            
            # Record the best loss
            if smoothed_loss < best_loss or batch_num==1:
                best_loss = smoothed_loss
            
            # Store the values
            losses.append(smoothed_loss)
            log_lrs.append(math.log10(lr))
            
            # Backward pass
            loss.backward()
            self.optimizer_G.step()
            
            # Update the lr for the next step and store
            lr *= mult
            self.optimizer_G.param_groups[0]['lr'] = lr
        
        return log_lrs, losses

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.G.load_state_dict(checkpoint['generator_state_dict'])
        self.D.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    def train(self, start_epoch, num_epochs):
        for epoch in tqdm(range(start_epoch, num_epochs)):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # Training step
            metrics = self.train_step(epoch)

            # Log metrics
            self.log_metrics(metrics)

            self.log_activation_stats()

            # Save checkpoint
            if epoch % 200 == 0:
                self.save_checkpoint(epoch, metrics)

            # LR Scheduler steps
            self.scheduler_G.step(metrics['loss_G'])
            self.scheduler_D.step(metrics['loss_D'])

            # Early stopping
            if self.early_stopping(metrics['loss_G'], self.G):
                print("Early stopping triggered.")
                break
            
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, metrics)

           # self.act_stats.plot_stats()
        self.act_stats.remove()

    def hook_fn(module, input, output):
        # Attach attributes to the module to store mean and std
        module.mean = output.data.mean().item()
        module.std = output.data.std().item()



import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNetGenerator, self).__init__()

        # Contracting path
        self.enc1 = self.conv_block(input_channels, 64, use_batchnorm=False)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512, use_dropout=True)

        # Expanding path
        self.dec1 = self.deconv_block(512, 256)
        self.dec2 = self.deconv_block(256*2, 128)
        self.dec3 = self.deconv_block(128*2, 64)
        self.final = nn.ConvTranspose2d(64*2, output_channels, kernel_size=4, stride=2, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=True, use_dropout=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def deconv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=True):
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm)]
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        dec1 = self.dec1(enc4)
        dec2 = self.dec2(torch.cat([dec1, enc3], 1))
        dec3 = self.dec3(torch.cat([dec2, enc2], 1))
        return torch.tanh(self.final(torch.cat([dec3, enc1], 1)))
    

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels):
        super(PatchGANDiscriminator, self).__init__()
        self.main = nn.Sequential(
            self.conv_block(input_channels, 64, use_batchnorm=False),
            self.conv_block(64, 128),
            self.conv_block(128, 256),
            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1)
        )

        self.initialize_weights()

    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        return self.main(x)


def init_weights_xavier(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def init_weights_he(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)

def init_weights_lsuv(model, dataloader):
    def get_variance(model, dataloader):
        model.eval()
        with torch.no_grad():
            for x, _ in dataloader:
                x = x.cuda()
                output = model(x)
                return output.var().item()

    for layer in model.children():
        if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

            variance = get_variance(model, dataloader)
            scale_factor = (1 / variance) ** 0.5
            layer.weight.data *= scale_factor
            layer.bias.data *= scale_factor
            

G = UNetGenerator(input_channels=1, output_channels=3)  # Assuming grayscale to RGB
D = PatchGANDiscriminator(input_channels=4)

optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

criterion_GAN = nn.BCEWithLogitsLoss()
criterion_L1 = nn.L1Loss()

trainer = Trainer(G, D, optimizer_G, optimizer_D, dataloader, device=device, lambda_L1=100, criterion_GAN=criterion_GAN, criterion_L1=criterion_L1, log_wandb=True, track_activations=True)

trainer.train(start_epoch=0, num_epochs=2000)



