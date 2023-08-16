from torch.nn import ReduceLROnPlateau

class Trainer:
   """
    Class to handle the training process of the Pix2Pix model.

    Attributes:
        G (torch.nn.Module): Generator model.
        D (torch.nn.Module): Discriminator model.
        optimizer_G (torch.optim.Optimizer): Optimizer for the generator.
        optimizer_D (torch.optim.Optimizer): Optimizer for the discriminator.
        device (torch.device): Device to run the model on (e.g., GPU).
        lambda_L1 (float): Weight for the L1 loss term.
        criterion_GAN (callable): Loss function for the GAN loss.
        criterion_L1 (callable): Loss function for the L1 loss.
        act_stats_gen (ActivationStats): Activation statistics tracker for the generator.
        act_stats_disc (ActivationStats): Activation statistics tracker for the discriminator.
        output_path (str): Path to save output files.
        callbacks (list): List of callback objects.
        dataloader_historical (torch.utils.data.DataLoader): DataLoader for historical black and white images.
        dataloader_modern_gray (torch.utils.data.DataLoader): DataLoader for modern grayscale images.
        scheduler_G (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler for the generator.
        scheduler_D (torch.optim.lr_scheduler.ReduceLROnPlateau): Learning rate scheduler for the discriminator.
        early_stopping (EarlyStopping): Early stopping callback.
        log_wandb (bool): Whether to log training information with WandB.
        track_activations (bool): Whether to track activations.
    """
    def __init__(self, G, D, optimizer_G, optimizer_D, dataloader_historical, dataloader_modern_gray, device, lambda_L1, criterion_GAN, criterion_L1, log_wandb=True, track_activations=False, output_path="."):
        self.G = G
        self.D = D
        self.optimizer_G = optimizer_G
        self.optimizer_D = optimizer_D
        #self.dataloader = dataloader
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
        self.dataloader_historical = dataloader_historical
        self.dataloader_modern_gray = dataloader_modern_gray
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
    """Plots activation statistics for the generator and discriminator."""
        self.activation_stats.plot_stats()

    def plot_color_dim(self):
      """Plots the color dimensions of the activations."""
        self.activation_stats.color_dim()

    def plot_dead_activations(self):
        self.activation_stats.dead_chart()

    def add_callback(self, callback):
      """
        Adds a callback to the trainer.

        Args:
            callback (Callback): Callback to add.
        """
        callback.trainer = self
        self.callbacks.append(callback)

    def luminance_loss(self, fake_color, bw_image):
      """
        Computes the luminance loss between the generated color image and the black and white image.

        Args:
            fake_color (torch.Tensor): Generated color image.
            bw_image (torch.Tensor): Original black and white image.

        Returns:
            torch.Tensor: Luminance loss.
        """
        # Convert the generated color image to grayscale
        fake_gray = torch.mean(fake_color, dim=1, keepdim=True)
        return self.criterion_L1(fake_gray, bw_image)
    
    def color_preservation_loss(self, fake_color, real_color):
      """
        Computes the color preservation loss between the generated color image and the real color image.

        Args:
            fake_color (torch.Tensor): Generated color image.
            real_color (torch.Tensor): Real color image.

        Returns:
            torch.Tensor: Color preservation loss.
        """
        return self.criterion_L1(fake_color, real_color)
    
    def weighted_L1_loss(self, fake_color, real_color, bw_image, alpha=0.8, beta=0.1, is_modern_gray=False):
      """
        Computes a weighted L1 loss that includes the standard L1 loss, luminance loss, and color preservation loss.

        Args:
            fake_color (torch.Tensor): Generated color image.
            real_color (torch.Tensor): Real color image.
            bw_image (torch.Tensor): Original black and white image.
            alpha (float, optional): Weight for the luminance loss. Defaults to 0.8.
            beta (float, optional): Weight for the color preservation loss. Defaults to 0.1.
            is_modern_gray (bool, optional): Whether the image is a modern grayscale image. Defaults to False.

        Returns:
            torch.Tensor: Weighted L1 loss.
        """
        # Standard L1 loss
        l1 = self.criterion_L1(fake_color, real_color)
        
        # Luminance loss
        lum_loss = self.luminance_loss(fake_color, bw_image)

        # Color preservation loss
        color_pres_loss = 0
        if is_modern_gray:
            color_pres_loss = self.color_preservation_loss(fake_color, real_color)
        
        return l1 + alpha * lum_loss + beta * color_pres_loss

    def train_step(self, epoch, mode='historical'):
       """
    Executes a single training step (epoch) for the generator and discriminator.

    Args:
        epoch (int): The current epoch number.
        mode (str): The mode of training - can be 'historical', 'modern_gray', or 'both'. Defaults to 'historical'.

    Returns:
        dict: A dictionary containing the average loss for the generator and discriminator for the epoch.
    """
        epoch_loss_G = 0
        epoch_loss_D = 0
        batch_idx = 0

        # Decide which dataloaders to use based on the mode
        if mode == 'historical':
            dataloaders = [self.dataloader_historical]
            total_batches = len(self.dataloader_historical)
        elif mode == 'modern_gray':
            dataloaders = [self.dataloader_modern_gray]
            total_batches = len(self.dataloader_modern_gray)
        elif mode == 'both':
            dataloaders = [self.dataloader_historical, self.dataloader_modern_gray]
            total_batches = len(self.dataloader_historical) + len(self.dataloader_modern_gray)
        else:
            raise ValueError(f"Invalid mode: {mode}. Allowed modes: ['historical', 'modern_gray', 'both']")

        total_batches = len(self.dataloader_historical) + len(self.dataloader_modern_gray)

        # Loop through both dataloaders sequentially
        for dataloader in dataloaders:
            is_modern_gray = (dataloader == self.dataloader_modern_gray)

            # Loop through the batches in the current dataloader
            for batch_idx, (bw_images, color_images) in enumerate(dataloader):
                if is_modern_gray:
                    print("Training on grayscale modern images.")
                else:
                    print("Training on historical images.")
                
                # Move data to device
                bw_images, color_images = bw_images.to(self.device), color_images.to(self.device)

                # generator forward pass
                fake_images = self.G(bw_images)

                if batch_idx % 10 == 0:  # Adjust this number to control the logging frequency
                    #unique_step = len(self.dataloader) * epoch + batch_idx  # Compute a unique step for every epoch and batch combination
                    self.log_images(bw_images, color_images, fake_images)

                batch_idx += 1

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
                loss_L1 = self.weighted_L1_loss(fake_images, color_images, bw_images, alpha=0.86, beta=0.05, is_modern_gray=is_modern_gray)
                loss_G = loss_GAN + self.lambda_L1 * loss_L1
                epoch_loss_G += loss_G.item()

                loss_G.backward()
                self.optimizer_G.step()

        self.log_activation_stats()
        print(f"Epoch: {epoch} | Batch: {batch_idx} | Loss: {epoch_loss_G / total_batches} | Loss: {epoch_loss_D / total_batches}")

        return {"loss_G": epoch_loss_G / total_batches, "loss_D": epoch_loss_D / total_batches}


    def log_metrics(self, metrics):
      """
    Logs the provided metrics to WandB if logging is enabled.

    Args:
        metrics (dict): A dictionary containing the metrics to log.
    """
        if self.log_wandb:
            wandb.log(metrics)

    def log_images(self, bw_images, color_images, fake_images):
       """
    Logs the black and white, real color, and generated color images to WandB.

    Args:
        bw_images (Tensor): Tensor of black and white images.
        color_images (Tensor): Tensor of real color images.
        fake_images (Tensor): Tensor of generated color images.
    """
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
      """
    Logs the activation statistics including mean and standard deviation for each layer of the generator and discriminator to WandB.
    """
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
      """
    Saves a checkpoint of the current model state, including epoch, model state dictionaries, optimizer state, and loss metrics.

    Args:
        epoch (int): The current epoch number.
        metrics (dict): A dictionary containing loss metrics.
    """
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
      """
    Finds the optimal learning rate using the learning rate finder technique.

    Args:
        init_value (float): The initial learning rate value. Defaults to 1e-8.
        final_value (float): The final learning rate value. Defaults to 10.0.
        beta (float): The smoothing factor for the loss. Defaults to 0.98.

    Returns:
        tuple: A tuple containing logged learning rates and corresponding losses.
    """
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
      """
    Loads a checkpoint from the given path and restores the model and optimizer states.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
    """
        checkpoint = torch.load(checkpoint_path)
        self.G.load_state_dict(checkpoint['generator_state_dict'])
        self.D.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")

    def train(self, start_epoch, num_epochs):
      """
    Executes the entire training process including training epochs, logging, checkpointing, and early stopping.

    Args:
        start_epoch (int): The starting epoch number.
        num_epochs (int): The total number of epochs to train.
    """
        for epoch in tqdm(range(start_epoch, num_epochs)):
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch)

            # Training step
            metrics = self.train_step(epoch)

            # Log metrics
            self.log_metrics(metrics)

            self.log_activation_stats()

            # Save checkpoint
            if epoch % 50 == 0:
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
      """
    A hook function that attaches attributes to the module to store mean and standard deviation of activations.

    Args:
        module (nn.Module): The module to which the hook is attached.
        input (tuple): The input to the module.
        output (Tensor): The output from the module.
    """
        # Attach attributes to the module to store mean and std
        module.mean = output.data.mean().item()
        module.std = output.data.std().item()
