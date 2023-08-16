import os
from PIL import Image
import torch.utils.data

class ColorizationDataset(torch.utils.data.Dataset):
   """
    A custom dataset class to handle the loading and preprocessing of historical black and white images 
    and their corresponding color images.

    Attributes:
        bw_dir (str): Directory containing the black and white images.
        color_dir (str): Directory containing the color images.
        use_modern_gray (bool): If True, uses modern grayscale images instead of historical ones.
        bw_transform (callable, optional): A function/transform to apply to the black and white images.
        color_transform (callable, optional): A function/transform to apply to the color images.
        image_files (list): List of image filenames in the dataset.
    """
  
    def __init__(self, bw_dir, color_dir, use_modern_gray=True, bw_transform=None, color_transform=None):
        self.bw_dir = bw_dir
        self.color_dir = color_dir
        self.use_modern_gray = use_modern_gray
        self.bw_transform = bw_transform
        self.color_transform = color_transform

        if self.use_modern_gray:
            self.image_files = [f for f in os.listdir(color_dir) if f.endswith('.jpg') or f.endswith('.jpeg')]
        else:
            self.image_files = [f for f in os.listdir(bw_dir) if f.endswith('.jpg') or f.endswith('.jpeg')]

        for img_file in self.image_files:
            bw_path = os.path.join(bw_dir, img_file)
            
            # Strip the file extension from the grayscale image's filename
            base_color_file = os.path.splitext(img_file)[0]
            
            # Determine the possible paths for the color image
            color_path1 = os.path.join(color_dir, base_color_file + '.jpeg')
            color_path2 = os.path.join(color_dir, base_color_file + '_result.jpeg')
            color_path3 = os.path.join(color_dir, base_color_file + '.jpg')
            
            assert os.path.exists(bw_path) and (os.path.exists(color_path1) or os.path.exists(color_path2) or os.path.exists(color_path3)), \
            f"Image pair not found for {img_file}. Checked paths: {bw_path} and {color_path1}/{color_path2}/{color_path3}"

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        bw_path = os.path.join(self.bw_dir, self.image_files[idx])
        
        # Determine the base color filename without the file extension
        base_color_file = os.path.splitext(self.image_files[idx])[0]
        
        # Determine the possible paths for the color image
        color_path1 = os.path.join(self.color_dir, base_color_file + '.jpeg')
        color_path2 = os.path.join(self.color_dir, base_color_file + '_result.jpeg')
        color_path3 = os.path.join(self.color_dir, base_color_file + '.jpg')
        
        # Decide which color path to use
        if os.path.exists(color_path1):
            color_path = color_path1
        elif os.path.exists(color_path2):
            color_path = color_path2
        elif os.path.exists(color_path3):
            color_path = color_path3
        else:
            raise ValueError(f'No matching color image found for {bw_path}. Checked paths: {color_path1}, {color_path2}, and {color_path3}')
        
        try:
            bw_image = Image.open(bw_path).convert("L")
            color_image = Image.open(color_path).convert("RGB")
        except Exception as e:
            raise ValueError(f'Error Loading images at {bw_path} or {color_path}. Exception: {e}')
        
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





def create_grayscale_dataset(color_dir, grayscale_dir):
     """
    Convert color images from the given directory to grayscale and save them to the target directory.

    Args:
        color_dir (str): Directory containing the original color images (JPG or JPEG format).
        grayscale_dir (str): Target directory where the grayscale images will be saved. If it doesn't exist, it will be created.

    Note:
        This function assumes that the color images are in JPG or JPEG format.
    """
    # Ensure the grayscale directory exists
    if not os.path.exists(grayscale_dir):
        os.makedirs(grayscale_dir)
    
    # Iterate through all color images and convert to grayscale
    for image_name in os.listdir(color_dir):
        if image_name.endswith('.jpg') or image_name.endswith('.jpeg'):
            color_image_path = os.path.join(color_dir, image_name)
            grayscale_image_path = os.path.join(grayscale_dir, image_name)
            
            # Open the color image and convert to grayscale
            with Image.open(color_image_path) as img:
                grayscale_img = img.convert("L")
                grayscale_img.save(grayscale_image_path)
