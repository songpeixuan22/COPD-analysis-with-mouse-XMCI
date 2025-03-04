import os
import imageio
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

class CustomDataset(Dataset):
    '''
    CustomDataset: A custom dataset class for loading images and labels
    '''
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.l_paths = []
        self.r_paths = []
        raw_path = os.path.join(data_dir, 'raw')
        mask_path = os.path.join(data_dir, 'mask')

        for filename in os.listdir(raw_path):
            if filename.endswith('.tif'):
                image_path = os.path.join(raw_path, filename)
                l_path = os.path.join(mask_path, 'l')
                r_path = os.path.join(mask_path, 'r')
                l_path = os.path.join(l_path, filename)
                r_path = os.path.join(r_path, filename)
                
                if os.path.exists(image_path) and os.path.exists(l_path) and os.path.exists(r_path):
                    self.image_paths.append(image_path)
                    self.l_paths.append(l_path)
                    self.r_paths.append(r_path)
                else:
                    print(f"Missing image or label: {image_path}, {l_path}, {r_path}")

        print("Data loaded")
        print(f"Found {len(self.image_paths)} images and {len(self.l_paths)} labels")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = imageio.imread(self.image_paths[idx])
        l = imageio.imread(self.l_paths[idx])
        r = imageio.imread(self.r_paths[idx])
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        # breakpoint()
        image, l, r = Image.fromarray(image), Image.fromarray(l), Image.fromarray(r)
        
        if self.transform:
            image = self.transform(image)
            l = self.transform(l)
            r = self.transform(r)
            image = normalize(image)
        else:
            transform = transforms.ToTensor()
            image = transform(image)
            l = transform(l)
            r = self.transform(r)

        label = np.stack([l.squeeze(), r.squeeze()], axis=0)
        # breakpoint()
        return image, label, self.image_paths[idx]

    def get_image_path(self, idx):
        return self.image_paths[idx]

    def get_label_path(self, idx):
        return self.l_paths[idx], self.r_paths[idx]

class CustomDataset2(Dataset):
    '''
    CustomDataset: A custom dataset class for loading images and labels
    '''
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        raw_path = os.path.join(data_dir, 'raw')

        for filename in os.listdir(raw_path):
            if filename.endswith('.tif'):
                image_path = os.path.join(raw_path, filename)
                
                if os.path.exists(image_path):
                    self.image_paths.append(image_path)
                else:
                    print(f"Missing image or label: {image_path}, {l_path}, {r_path}")

        print("Data loaded")
        print(f"Found {len(self.image_paths)} images")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = imageio.imread(self.image_paths[idx])
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
        # breakpoint()
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
            image = normalize(image)
        else:
            transform = transforms.ToTensor()
            image = transform(image)

        # breakpoint()
        return image, self.image_paths[idx]

    def get_image_path(self, idx):
        return self.image_paths[idx]