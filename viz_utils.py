class Visualizer:
  """
    A class to handle the visualization of black and white images, color images, and generated color images.

    Attributes:
        unnormalize (bool): Whether to unnormalize the images before displaying.
    """
    def __init__(self, unnormalize=True):
        self.unnormalize = unnormalize

    def _unnormalize(self, img):
       """
        Unnormalizes the given image tensor if self.unnormalize is True.

        Args:
            img (torch.Tensor): Image tensor to unnormalize.

        Returns:
            torch.Tensor: Unnormalized image tensor.
        """
        if self.unnormalize:
            img = img * 0.5 + 0.5
        return img.clamp(0, 1)
    
    def display_images(self, bw_images, color_images, nrows=3):
      """
        Displays the black and white images and corresponding color images in a grid.

        Args:
            bw_images (torch.Tensor): Tensor containing black and white images.
            color_images (torch.Tensor): Tensor containing color images.
            nrows (int, optional): Number of rows in the grid. Defaults to 3.
        """
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
      """
        Displays the black and white images, ground truth color images, and generated color images in a grid.

        Args:
            bw_images (torch.Tensor): Tensor containing black and white images.
            color_images (torch.Tensor): Tensor containing ground truth color images.
            generated_images (torch.Tensor): Tensor containing generated color images.
            nrows (int, optional): Number of rows in the grid. Defaults to 3.
        """
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
