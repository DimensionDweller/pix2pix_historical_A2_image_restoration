import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
   """
    Implements a U-Net based generator model.

    Args:
        input_channels (int): Number of input channels.
        output_channels (int): Number of output channels.
    """
    def __init__(self, input_channels, output_channels):
        super(UNetGenerator, self).__init__()

        # Contracting path
        self.enc1 = self.conv_block(input_channels, 64, use_batchnorm=False)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.enc5 = self.conv_block(512, 512, use_dropout=True)
        self.enc6 = self.conv_block(512, 512, use_dropout=True)

        # Expanding path
        self.dec1 = self.deconv_block(512, 512)
        self.dec2 = self.deconv_block(512*2, 512)
        self.dec3 = self.deconv_block(512*2, 512)
        self.dec4 = self.deconv_block(512 + 256, 512)  # Adjusted this line
        self.dec5 = self.deconv_block(512 + 128, 256)  # Adjusted this line to keep the pattern
        self.dec6 = self.deconv_block(256 + 64, 128)   # Adjusted this line to keep the pattern
        self.final = nn.ConvTranspose2d(128, output_channels, kernel_size=4, stride=2, padding=1)
        self.downsample = nn.Conv2d(3, output_channels, kernel_size=4, stride=2, padding=1)

        self.initialize_weights()

    def initialize_weights(self):
      """
        Initializes the weights of the convolutional layers using He initialization.
        """
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=True, use_dropout=False):
      """
        Constructs a convolutional block with optional batch normalization and dropout.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel. Defaults to 4.
            stride (int): Stride of the convolution. Defaults to 2.
            padding (int): Padding for the convolution. Defaults to 1.
            use_batchnorm (bool): Whether to include a batch normalization layer. Defaults to True.
            use_dropout (bool): Whether to include a dropout layer. Defaults to False.

        Returns:
            nn.Sequential: A sequential container of the layers forming the block.
        """
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def deconv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=True):
      """
        Constructs a deconvolutional block with optional batch normalization.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the deconvolutional kernel. Defaults to 4.
            stride (int): Stride of the deconvolution. Defaults to 2.
            padding (int): Padding for the deconvolution. Defaults to 1.
            use_batchnorm (bool): Whether to include a batch normalization layer. Defaults to True.

        Returns:
            nn.Sequential: A sequential container of the layers forming the block.
        """
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        return nn.Sequential(*layers)

    def forward(self, x):
       """
        Forward pass through the generator.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The generated output tensor.
        """
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)

        dec1 = self.dec1(enc6)
       # print(dec1.shape)
        dec2 = self.dec2(torch.cat([dec1, enc5], 1))
       # print(dec2.shape)
        dec3 = self.dec3(torch.cat([dec2, enc4], 1))
       # print(dec3.shape)
        dec4 = self.dec4(torch.cat([dec3, enc3], 1))
        #print(dec4.shape)
        dec5 = self.dec5(torch.cat([dec4, enc2], 1))
       # print(dec5.shape)
        dec6 = self.dec6(torch.cat([dec5, enc1], 1))
       # print(dec6.shape)
        output = torch.tanh(self.final(dec6))
        
        return self.downsample(output)
            

class PatchGANDiscriminator(nn.Module):
  """
    Implements a PatchGAN discriminator model.

    Args:
        input_channels (int): Number of input channels.
    """
    def __init__(self, input_channels):
        super(PatchGANDiscriminator, self).__init__()

        # Modified layers for the discriminator
        self.main = nn.Sequential(
            self.conv_block(input_channels, 32, use_batchnorm=False), 
            self.conv_block(32, 64),
            self.conv_block(64, 128),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1)  # Adjusted this layer
        )

        self.initialize_weights()

    def conv_block(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_batchnorm=True, use_dropout=True):
      """
        Constructs a convolutional block with optional batch normalization and dropout.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel. Defaults to 4.
            stride (int): Stride of the convolution. Defaults to 2.
            padding (int): Padding for the convolution. Defaults to 1.
            use_batchnorm (bool): Whether to include a batch normalization layer. Defaults to True.
            use_dropout (bool): Whether to include a dropout layer. Defaults to False.

        Returns:
            nn.Sequential: A sequential container of the layers forming the block.
        """
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding_mode='reflect', bias=not use_batchnorm)]
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if use_dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)
    
    def initialize_weights(self):
      """
        Initializes the weights of the convolutional layers using He initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
      """
        Forward pass through the discriminator.

        Args:
            x (Tensor): The input tensor.

        Returns:
            Tensor: The discriminator's output tensor.
        """
        return self.main(x)
