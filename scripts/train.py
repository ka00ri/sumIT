import os
import argparse
import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Disable GPU support 
os.environ["CUDA_VISIBLE_DEVICES"]=""

import torch
import torchvision
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

# Run on CPU
device = torch.device("cpu")

# Local imports
from dataset import DataSetSum, DataSetDiffSum
from model import SuMNISTNet, DiffSuMNISTNet
from utils import construct_operations, show

# Set seed
torch.manual_seed(42)


#===========================#
# Trains and evaluates Task 1
def train_eval_sum(model, 
                   dataset,
                   writer,
                   device,
                   args):

    # Set paths
    models_path = Path('./models/SuMNIST')

    # Prepare train and validation datasets
    # Split data into training (80%) and validation datasets (20%)
    N = len(dataset)
    n_validation = int(N * 0.2)
    n_train = N - n_validation
    train_dataset, validation_dataset = random_split(dataset, [n_train, n_validation], 
    generator=torch.Generator().manual_seed(42))

    # Load data
    loader_args = dict(batch_size=args.batch_size, num_workers= 4)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **loader_args)
    validation_loader = DataLoader(dataset=validation_dataset, shuffle=False,drop_last=True, **loader_args)
    
    # Set up criterion and metric 
    criterion = nn.MSELoss()
    mae = nn.L1Loss()
    min_validation_loss = np.inf 

    # Set up optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate, rho=0.95, eps=1e-08)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    
    # Comment out to use lr scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

    # Load from checkpoint if not None
    epoch_at = 0
    if args.restore_from is not None:
        checkpoint = torch.load(args.restore_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_at = checkpoint['epoch']

    # Train and validation loop
    for epoch in range(epoch_at+1, args.epochs+1):
        # TRAINING
        model.train()
        train_loss = 0.0
        train_floored = 0
        train_ceiled = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch/args.epochs}', unit='img') as pbar:
            for data_dict in train_loader:

                data = data_dict['image'].to(device)
                target = torch.unsqueeze(data_dict['label'], 1).to(device)     
                
                # Clear the gradient: per default accumulative
                optimizer.zero_grad()

                # Forward pass
                output = model(data[:, :, :, :28], data[:, :, :, 28:])

                # Calc MSE loss
                loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()
                # scheduler.step()

                # Update pbar
                pbar.update(data.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Calcculate loss
                train_loss+=loss.item()*data.size(0)
        
                # Metrics
                train_floored+=mae(torch.floor(output), target).item()*data.size(0)
                train_ceiled+=mae(torch.ceil(output), target).item()*data.size(0)

                
        # tensorboard
        writer.add_scalar('Training loss', train_loss/n_train, epoch)
        writer.add_scalar('Train/ MAE ceil', train_floored/n_train, epoch)
        writer.add_scalar('Train/ MAE floor', train_ceiled/n_train, epoch)


        # Evaluation
        model.eval()
        validation_loss = 0.0
        validation_floored = 0
        validation_ceiled = 0
        with torch.no_grad():       
            for data_dict in validation_loader:     
                data = data_dict['image'].to(device)
                target = torch.unsqueeze(data_dict['label'], 1).to(device)    

                # Forward pass
                output = model(data[:, :, :, :28], data[:, :, :, 28:])

                # Calc MSE loss
                loss = criterion(output, target)
                validation_loss += loss.item()*data.size(0)

                # Metrics
                validation_floored+=mae(torch.floor(output), target).item()*data.size(0)
                validation_ceiled+=mae(torch.ceil(output), target).item()*data.size(0)

        print(f"Epoch: {epoch+1}/{args.epochs}.. ", 
        f"Training Loss: {train_loss/n_train:.6f}.. ",
        f"Validation Loss: {validation_loss/n_validation:.6f}.. ")

        # Write per Epoch
        writer.add_scalar('Validation loss', validation_loss/n_validation, epoch)
        writer.add_scalar('Val/MAE floor', validation_floored/n_validation, epoch)
        writer.add_scalar('Val/MAE ceil', validation_ceiled/n_validation, epoch)

        # Save/Update best model 
        Path(models_path).mkdir(parents=True, exist_ok=True)
        if validation_loss < min_validation_loss:
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss},
                        os.path.join(models_path,  f'epoch{epoch}.pth'))
            min_validation_loss = validation_loss

    return model


#============================#
# Trains and evaluates Task 2
def train_eval_diffsum(model, 
                       dataset,
                       writer,
                       device,
                       args):

    # Set paths
    models_path = Path('./models/DiffSuMNIST')

    # Prepare train and validation datasets
    # Split data into training (80%) and validation dataset (20%)
    N = len(dataset)
    n_validation = int(N * 0.2)
    n_train = N - n_validation
    train_dataset, validation_dataset = random_split(dataset, [n_train, n_validation], 
    generator=torch.Generator().manual_seed(42))

    # Load data
    loader_args = dict(batch_size=args.batch_size, num_workers= 4, drop_last=True)
    train_loader = DataLoader(dataset=train_dataset, shuffle=True, **loader_args)
    validation_loader = DataLoader(dataset=validation_dataset, shuffle=False,  **loader_args)
    

    # Set criterion and metric 
    criterion = nn.MSELoss()
    mae = nn.L1Loss()
    min_validation_loss = np.inf

    # Set up optimizer
    optimizer = optim.Adadelta(model.parameters(), lr=args.learning_rate, rho=0.95, eps=1e-08)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


    # Load from checkpoint if not None
    epoch_at = 0
    if args.restore_from is not None:
        checkpoint = torch.load(args.restore_from)
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch_at = checkpoint['epoch']

    # Get operations
    operations = construct_operations() 

    # Train and validation loop
    for epoch in range(epoch_at+1, args.epochs+1):
        # TRAINING
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        with tqdm(total=n_train, desc=f'Epoch {epoch/args.epochs}', unit='img') as pbar:
            for data_dict in train_loader:

                data = data_dict['image'].to(device)
                target_sum = torch.unsqueeze(data_dict['label_sum'], 1).to(device)   
                target_diff = torch.unsqueeze(data_dict['label_diff'], 1).to(device)
                
                # Clear the gradient: per default accumulative
                optimizer.zero_grad()

                # sample operation from a uniform distribution
                idx = np.random.randint(0, 2, 1, dtype=np.uint8)
                op = operations[idx[0]]
                op_torch = torch.from_numpy(op).to(device)
                op_torch = op_torch.expand(args.batch_size, -1,-1,-1)

                # Forward pass
                output = model(data[:, :, :, :28], data[:, :, :, 28:],  op_torch)

                target = None
                # select correct target and calc MSE loss
                if idx[0] == 0:
                    target = target_sum
                else:
                    target = target_diff

                loss = criterion(output, target)

                # Backward pass
                loss.backward()

                # Update weights
                optimizer.step()
                # scheduler.step()

                # Update pbar
                pbar.update(data.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Calcculate loss
                train_loss+=loss.item()*data.size(0)

                # Metrics
                train_mae+=mae(output, target).item()*data.size(0)
                
        # tensorboard
        writer.add_scalar('Training loss', train_loss/n_train, epoch)
        writer.add_scalar('Train/ MAE ', train_mae/n_train, epoch)


        # Evaluation
        model.eval()
        validation_loss = 0.0
        validation_mae = 0.0
        with torch.no_grad():       
            for data_dict in validation_loader:     
                data = data_dict['image'].to(device)
                target_sum = torch.unsqueeze(data_dict['label_sum'], 1).to(device)   
                target_diff = torch.unsqueeze(data_dict['label_diff'], 1).to(device)   

                idx = np.random.randint(0, 2, 1, dtype=np.uint8)
                op = operations[idx[0]]
                op_torch = torch.from_numpy(op).to(device)
                op_torch = op_torch.expand(args.batch_size, -1,-1,-1)

                # Forward pass
                output = model(data[:, :, :, :28], data[:, :, :, 28:], op_torch)

                # select correct target and calc MSE loss
                if idx == 0:
                    loss = criterion(output, target_sum)
                else:
                    loss = criterion(output, target_diff)

                validation_loss += loss.item()*data.size(0)

                # Metrics
                validation_mae+=mae(output, target).item()*data.size(0)

        print(f"Epoch: {epoch+1}/{args.epochs}.. ", 
              f"Training Loss: {train_loss/n_train:.6f}.. ",
              f"Validation Loss: {validation_mae/n_validation:.6f}.. ")

        # Write per Epoch
        writer.add_scalar('Validation loss', validation_loss/n_validation, epoch)
        writer.add_scalar('Val/MAE ', validation_mae/n_validation, epoch)

        # Save/Update best model 
        Path(models_path).mkdir(parents=True, exist_ok=True)
        if validation_loss < min_validation_loss:
            torch.save({'epoch': epoch, 
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss},
                        os.path.join(models_path,  f'epoch{epoch}.pth'))
            min_validation_loss = validation_loss

    return model
    

def get_args():
    # pars arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-dt', '--train_dir', default='data/train', help='directry to read train data from')
    parser.add_argument('-ds', '--test_dir', default='data/test', help='directry to read test data from')
    parser.add_argument('-e', '--exp', default='exp', type=str, help='experiment name/description')
    parser.add_argument('-bs', '--batch_size', default=128, type=int,  help='Default 128')
    parser.add_argument('-ep', '--epochs', default=50, type=int,  help='Default 50')
    parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='Default 1e-2')
    parser.add_argument('-m', '--model', default="SuMNIST", type=str, help='choose model to run: SuMNIST or DiffSuMNIST')
    parser.add_argument('-r', '--restore_from', default=None,
                        help='path to restore from, e.g. ./models/....pth')
    parser.add_argument('-dp', '--display_sample', default=False,
                        help='set to True to display a data sample')

    return parser.parse_args()


# Main function
def main():
    args = get_args()

    # Data absolute paths
    path = os.path.dirname(os.path.realpath(__file__))
    train_path = os.path.join(path, args.train_dir)
    test_path = os.path.join(path, args.test_dir)

    # Writer will output to ./runs/ directory by default
    run_path = os.path.join(path, f"runs/{args.model}/exp_{args.exp}_{args.learning_rate}_{args.epochs}")
    writer = SummaryWriter(run_path)

    # Saved models directory
    saved_model_path = os.path.join(path, 'models')
    if not os.path.exists(saved_model_path):
        os.makedirs(os.path.join(saved_model_path))

    # Set up transformations
    train_transform = transforms.Compose([transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2),
                    transforms.ToTensor()])

    test_transform = transforms.Compose([transforms.ToTensor()])

    # Test dataset and loader
    test_dataset = DataSetSum(files_dir=test_path, transform=test_transform)
    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=args.batch_size,
                            drop_last=False,
                            num_workers=4,
                            shuffle=True)      

    # Display sample
    samples = iter(test_loader).next()
    sample = samples['image'][0]
    if args.display_sample:
        for key, value in samples:
            if 'label' in key:
                print(f"{key}: ", value[0].item())
        plt.imshow(torch.squeeze(sample), cmap='gray')
        plt.show()

    img_grid = torchvision.utils.make_grid(samples['image'], nrow=16)
    # show(img_grid)

    # save sample to tensorboard
    writer.add_image('mnist_images', img_grid)

    # Choose dataset, dataloader and fucntion to run given model name: 'SuMNIST' or 'DiffSuMNIST'
    dataset, model = None, None
    # Choose dataset and model to run
    if args.model == 'SuMNIST':
        dataset = DataSetSum(files_dir=train_path, transform=train_transform)
        
        model = SuMNISTNet().to(device)

        print(f"Model {args.model} summary: ")
        summary(model, [(1, 28, 28), (1, 28, 28)])

        # save graph to tensorboard
        writer.add_graph(model, (samples['image'][:, :, :, :28],
                                 samples['image'][:, :, :, :28]))

        # Run training/evaluation loop
        model = train_eval_sum(model=model, 
        dataset=dataset, device=device, writer=writer, args=args)

    elif args.model == 'DiffSuMNIST':
        dataset = DataSetDiffSum(files_dir=train_path, transform=train_transform)

        model = DiffSuMNISTNet().to(device)
        print(f"Model {args.model} summary: " )
        summary(model, [(1, 28, 28), (1, 28, 28), (1, 28, 28)])
    
        # save graph to tensorboard
        writer.add_graph(model, [samples['image'][:, :, :, :28], 
                                 samples['image'][:, :, :, :28], 
                                 samples['image'][:, :, :, :28]])

        # Run training/evaluation loop
        model = train_eval_diffsum(model=model, 
                                  dataset=dataset,
                                  device=device, 
                                  writer=writer, 
                                  args=args)
    else:
        raise ValueError('Only SuMNIST and DiffSuMNIST models are supported!')

    print("Training done!")

    # Close tensorboard writer
    writer.close()

  
if __name__ == "__main__":
    main()