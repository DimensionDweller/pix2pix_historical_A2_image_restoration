class Callback:
  """
    Base class for implementing custom callbacks during training.
    Can be used to define actions at various stages of training (e.g., start/end of epoch or batch).
    """
    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass


class ModelCheckpoint(Callback):
  """
    Callback to save the model during training.

    Attributes:
        filepath (str): Path to save the model.
        monitor (str): Metric to monitor for saving the best model.
        save_best_only (bool): If True, only save the model if it's the best so far.
    """
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
            self.best = 

class EarlyStopping:
  """
    Callback to stop training early if no improvement is observed.

    Attributes:
        patience (int): Number of epochs with no improvement to wait before stopping.
        verbose (bool): Whether to print messages.
        delta (float): Minimum change in monitored value to qualify as an improvement.
    """
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
   """
    Base class to register hooks to PyTorch modules.

    Attributes:
        hooks (list): List of registered hooks.
    """
    def __init__(self, hook_fn, modules):
        self.hooks = []
        for module in modules:
            self.hooks.append(module.register_forward_hook(hook_fn))

    def remove(self):
        for hook in self.hooks:
            hook.remove()


class ActivationStats(HooksCallback):
  """
    Callback to collect and visualize statistics of activations during training.
    Extends the HooksCallback class to provide specific functionality for activation statistics.
    """
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
