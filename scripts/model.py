from nntplib import NNTP
import torch
import torch.nn as nn


class SuMNISTNet(nn.Module):
    """ Network to add digits of two images."""

    def __init__(self,):
        super().__init__()

        # CNN: extract features per image 
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25))
        
        # MLP: combine features of both images
        self.mlp = nn.Sequential(
            nn.Linear(128 * 12 * 12, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(256, 1),
        )

    def forward(self, x_1, x_2):
        x_1 = self.conv_block(x_1)
        x_2 = self.conv_block(x_2)
        x = torch.concat((x_1, x_2), axis=1)  # concat along channel axis
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x


class DiffSuMNISTNet(nn.Module):
    """ Network to add and substract digits of two images."""

    def __init__(self,):
        super().__init__()

        # CNN: extract features per image 
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25))
        
        # MLP: combine features of all inputs
        self.mlp = nn.Sequential(
            nn.Linear(192 * 12 * 12, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(1024, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

    def forward(self, x_1, x_2, op):
        x_1 = self.conv_block(x_1)
        x_2 = self.conv_block(x_2)
        op = self.conv_block(op)
        x = torch.concat([x_1, x_2, op], axis=1) # concat along channel axis
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x
    

class SuMNISTNet2(nn.Module):
    """ Network to add digits of two images."""
    def __init__(self,):
        super().__init__()

        # CNN: extract features per image 
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25)
        )
        
        # NN: combine features of both images
        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 1)
        )

    def forward(self, x_1, x_2):
        x_1 = self.conv1(x_1)
        x_1 = self.conv2(x_1)
        x_1 = self.conv3(x_1)

        x_2 = self.conv1(x_2)
        x_2 = self.conv2(x_2)
        x_2 = self.conv3(x_2)

        x = torch.concat((x_1, x_2), axis=1) # concat along channel axis
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x
    










