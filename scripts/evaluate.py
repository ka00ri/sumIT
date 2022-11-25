from operator import ge
import os
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from dataset import DataSetSum, DataSetDiffSum
from model import SuMNISTNet, DiffSuMNISTNet
from utils import construct_operations


def evaluate_sum(data_path, model_path, batch_size=1):
    ''' Evaluates a model on a test dataset by calculating the 
        sum of correctly and incorrectly predicted labels.'''

    # Run on CPU
    device = torch.device("cpu")

    # Load test dataset
    test_transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = None
    model = None

    # Get test dataset
    test_dataset = DataSetSum(files_dir=data_path, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                    batch_size=batch_size,
                    drop_last=False,
                    num_workers=4,
                    shuffle=True)     

    
    # Initialize and load model
    model = SuMNISTNet().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    #model.load_state_dict(torch.load(model_path))
    model.eval()

    num_correct = 0
    num_incorrect = 0

    for data_dict in test_loader:

        data = data_dict['image'].to(device)
        target = torch.unsqueeze(data_dict['label'], 1).to(device) 
        output = model(data[:, :, :, :28], data[:, :, :, 28:])

        rounded = torch.round(output)

        if abs(rounded - target) <= 1:
            num_correct +=1
        else:
            num_incorrect +=1

    print("Total number of test samples: ", len(test_dataset))
    print("Total number of correctly predicted sum of image pairs: ", num_correct)
    print("Total number of incorrectly predicted sum of image pairs: ", num_incorrect)
    return


def evaluate_diffsum(data_path, model_path, batch_size=1):
    ''' Evaluates a model on a test dataset by calculating the 
    sum of correctly and incorrectly predicted labels.'''

    # Run on CPU
    device = torch.device("cpu")

    # Load test dataset
    test_transform = transforms.Compose([transforms.ToTensor()])

    test_dataset = None
    model = None

    # Get test dataset
    test_dataset = DataSetDiffSum(files_dir=data_path, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                    batch_size=batch_size,
                    drop_last=False,
                    num_workers=4,
                    shuffle=True)     

    # Initialize and load model
    model = DiffSuMNISTNet().to(device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Operations input for DiffSuMNIST model
    operations = construct_operations()
    
    # sample operation from a uniform distribution
    idx = np.random.randint(0, 2, 1, dtype=np.uint8)
    op = operations[idx[0]]
    op_torch = torch.from_numpy(op).to(device)
    op_torch = op_torch.expand(batch_size, -1,-1,-1)

    num_correct = 0
    num_incorrect = 0
    num_incorrect_sign = 0

    for data_dict in test_loader:

        data = data_dict['image'].to(device)
        target_sum = torch.unsqueeze(data_dict['label_sum'], 1).to(device)   
        target_diff = torch.unsqueeze(data_dict['label_diff'], 1).to(device)

        output = model(data[:, :, :, :28], data[:, :, :, 28:], op_torch)

        target = None
        # select correct target and calc MSE loss
        if idx[0] == 0:
            target = target_sum
        else:
            target = target_diff

        rounded = torch.round(output)

        num_incorrect_sign += (target * output) > 0

        if abs(rounded - target) <= 1:
            num_correct +=1
        else:
            num_incorrect +=1

    print("Total number of test samples: ", len(test_dataset))
    print("Total number of correctly predicted sum of image pairs: ", num_correct)
    print("Total number of incorrectly predicted sum of image pairs: ", num_incorrect)
    print("Total number of incorrect signs between target and prediction: ", num_incorrect_sign.item())
    return

def get_args():
    # pars arguments
    parser = argparse.ArgumentParser() 
    parser.add_argument('-mn', '--model_name', default='SuMNIST', help='choose model to run: SuMNIST or DiffSuMNIST')
    parser.add_argument('-dp', '--data_path', default='./data/test', help='directry to laod models from')
    parser.add_argument('-mp', '--model_path', default='./models', help='directry to laod models from')
    parser.add_argument('-bs', '--batch_size', default=1, type=int,  help='Default 1')
    return parser.parse_args()

  
# Main
if __name__ == "__main__":
    args = get_args()
    
    path = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(path, 'models/DiffSuMNIST/epoch22.pth') # sample 
    data_path = os.path.join(path, 'data/test')

    
    if not os.path.isfile(model_path):
        print("Invalid model file!")
    
    if not os.path.exists(data_path):
        print("Invalid data path!")
        
    if args.model_name == 'SuMNIST':
        evaluate_sum(data_path, model_path)
    elif args.model_name == 'DiffSuMNIST':
        evaluate_diffsum(data_path, model_path)
    else:
        raise ValueError('Only SuMNIST and DiffSuMNIST models are supported!')
    