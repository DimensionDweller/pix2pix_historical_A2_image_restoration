# Historical Image Colorization using UNet Generator and PatchGAN Discriminator

This project presents a novel approach to colorize grayscale images, employing a UNet Generator and PatchGAN Discriminator. Trained on both historical and modern grayscale images, the model can generate colorized versions of historical images of houses around Ann Arbor, Michigan, preserving intricate textures and details.

## Table of Contents

- [Background](#background)
- [Project Description](#project-description)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [Sources](#sources)

## Background

Grayscale image colorization has been a fascinating yet challenging problem in computer vision. Traditional methods relied on hand-crafted features and specific color mapping techniques. Recently, deep learning and, particularly, Generative Adversarial Networks (GANs) have shown promising results in this domain. This project implements a robust architecture using a UNet Generator and PatchGAN Discriminator, exploring unique losses and techniques to provide realistic colorization.

## Project Description

The aim of this project is to develop a deep learning model capable of colorizing grayscale images while preserving original textures, luminance, and details. Special emphasis is placed on handling different types of grayscale images such as historical photographs and modern grayscale pictures.

This project includes:

- **Dual Data Handling**: Ability to train on both historical grayscale images and modern grayscale images, providing diverse training scenarios.
- **Unique Loss Functions**: A combination of standard GAN loss with luminance and color preservation losses to ensure realistic colorization.
- **Activation Statistics Tracking**: A feature to track activation statistics across different layers for insights into the learning dynamics of the model.
- **WandB Integration**: Integration with Weights & Biases for seamless logging and tracking.

### Dataset Overview

The model is trained on a combination of historical grayscale images and modern grayscale images where true color is known. Key characteristics of the dataset include:

- **Content**: Varied set of images including historical photographs and modern images.
- **Resolution**: Standardized dimensions for uniformity in training.
- **Preprocessing**: Techniques like normalization and augmentation to enhance the learning process.
- **Use Case Alignment**: The diverse nature of the dataset enables the model to learn a wide array of colorization techniques, catering to different scenarios.

## Model Architecture

### UNet Generator

The UNet Generator is designed to retain spatial information, allowing high-quality colorization. It comprises both contracting and expanding paths:

- **Contracting Path**: Consists of convolutional blocks, reducing dimensions while capturing features.
- **Expanding Path**: Utilizes deconvolutional blocks, upscaling the features to construct the colorized image.

### PatchGAN Discriminator

The PatchGAN Discriminator focuses on local patches, making it sensitive to textures and details. It uses a series of convolutional layers to classify real and fake colorized images.

## Loss Function

The model leverages a unique combination of losses to achieve realistic and high-quality colorization. The selection of these specific loss functions is crucial in capturing intricate details, preserving original textures, and maintaining color properties in the transformed images.

### GAN Loss

The GAN loss is a fundamental component in training the Generative Adversarial Network. It consists of two parts: the Discriminator loss and the Generator loss.

1. **Discriminator Loss**: It aims to classify the real and generated images correctly.

   $\[
   L_{\text{Discriminator}} = \mathbb{E}_{x \sim P_{\text{real}}}[\log D(x)] + \mathbb{E}_{z \sim P_{z}}[\log (1 - D(G(z)))]
   \]$

   where \( x \) are the real images, \( z \) are the noise samples, \( D \) is the Discriminator, and \( G \) is the Generator.

2. **Generator Loss**: It guides the Generator to produce images that the Discriminator classifies as real.

   \[
   L_{\text{Generator}} = \mathbb{E}_{z \sim P_{z}}[\log (1 - D(G(z)))]
   \]

The GAN loss ensures that the generated images are not only realistic but also indistinguishable from real images.

### Weighted L1 Loss

The weighted L1 loss is a specially designed loss for this project that combines different aspects of image quality.

1. **Standard L1 Loss**: Measures the absolute difference between the generated image and the real image. It ensures that the generated image closely resembles the real one.

   \[
   L_{\text{L1}} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|
   \]

   where \( y_i \) is the real image and \( \hat{y}_i \) is the generated image.

2. **Luminance Preservation Loss**: Ensures that the brightness and contrast of the generated image match the original grayscale image.

   \[
   L_{\text{luminance}} = \frac{1}{N} \sum_{i=1}^{N} \left( \text{Y}(y_i) - \text{Y}(\hat{y}_i) \right)^2
   \]

   where \( \text{Y} \) represents the luminance channel.

3. **Color Preservation Loss**: Ensures that the color hues of the generated image are consistent with real color hues, preserving the natural appearance.

   \[
   L_{\text{color}} = \frac{1}{N} \sum_{i=1}^{N} \left( \text{CbCr}(y_i) - \text{CbCr}(\hat{y}_i) \right)^2
   \]

   where \( \text{CbCr} \) represents the chrominance channels.

The total weighted L1 loss is a combination of these components:

\[
L_{\text{weighted\_L1}} = L_{\text{L1}} + \lambda_{\text{luminance}} L_{\text{luminance}} + \lambda_{\text{color}} L_{\text{color}}
\]

where \( \lambda_{\text{luminance}} \) and \( \lambda_{\text{color}} \) are the weighting factors.

### Conclusion on Loss Functions

The intricate design of these loss functions aids in achieving high-quality colorization. The GAN loss ensures realistic rendering, while the weighted L1 loss focuses on maintaining the essence of the original image, including luminance and color properties. This combination of losses is essential for achieving the desired balance between realism and authenticity in the generated images, making it tailored for this specific project.

## Results

(Include some generated images or a link to a demonstration, showcasing the results.)

## Usage

Clone the repository:

```bash
git clone https://github.com/YourUsername/ImageColorization.git
```

Navigate to the directory:

```bash
cd ImageColorization
```

Install required packages:

```bash
pip install -r requirements.txt
```

Run the training script:

```bash
python main.py
```

You can customize hyperparameters in the `main.py` file.

## Future Work and Conclusion

This project presents a novel approach to grayscale image colorization using deep learning, demonstrating promising results. The unique combination of the UNet Generator, PatchGAN Discriminator, and specialized loss functions offers a robust model capable of handling various grayscale images.

Future work may include:

- Experimenting with different architectures and loss functions.
- Extending the model to other domains such as artistic stylization.

The provided code serves as a solid foundation for anyone interested in diving into the fascinating world of image colorization. With continuous refinements and exploration, this model can be further enhanced to achieve state-of-the-art performance.

## Sources

(Include any relevant papers, links, or acknowledgments related to the project.)

---

Feel free to modify or add any sections as needed. Let me know if there are any specific details you would like to include!
