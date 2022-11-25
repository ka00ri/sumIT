from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class DataSetSum(Dataset):
    """ Custom dataset loads image and constructs label given image name."""
    def __init__(self, files_dir: str, transform=None):
        
        self.files_dir = Path(files_dir)
        self.samples = [file for file in Path(self.files_dir).glob('**/*.png')]
        if not self.samples:
            raise RuntimeError(f'No valid input files found in {self.files_dir}')
        print(f'Found dataset with {len(self.samples)} samples')

        # Transformation on image level
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path = self.samples[idx]
        image = Image.open(image_path)
        
        # Get label from file name
        # File name format: ../data/test/8_6/8_6_8.png
        sum_to_calc = self.samples[idx].name.split('_')[0:2]
        label = sum(map(int, sum_to_calc))

        return {'image': self.transform(image).type(torch.float32),
            'label': torch.as_tensor(label).type(torch.float32)
        }


class DataSetDiffSum(Dataset):
    """ Custom dataset loads image and constructs labels given image name."""
    def __init__(self, files_dir: str, transform=None):
        
        self.files_dir = Path(files_dir)
        self.samples = [file for file in Path(self.files_dir).glob('**/*.png')]
        if not self.samples:
            raise RuntimeError(f'No valid input files found in {self.files_dir}')
            
        print(f'Found dataset with {len(self.samples)} samples')

        # Transformation on image level
        self.transform = transform

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        image_path = self.samples[idx]
        image = Image.open(image_path)
        
        # Get labels from file name
        # File name format: ../data/test/8_6/8_6_8.png
        classes = self.samples[idx].name.split('_')[0:2]
        label_sum = int(classes[0]) + int(classes[1])
        label_diff = int(classes[0]) - int(classes[1])

        
        return {'image': self.transform(image).type(torch.float32),
            'label_sum': torch.as_tensor(label_sum).type(torch.float32), 
            'label_diff': torch.as_tensor(label_diff).type(torch.float32)
        }