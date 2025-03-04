#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_unet.py
~~~~~~~~~~~~~~~~~~~~~
This script trains an U-Net to segment the lung picture(two channels, left and right).
You can choose the type of net in utils.network.
model is saved into 'model.pth'.

:author: Peixuan Song
:email: spx22@mails.tsinghua.edu.cn
:copyright: (c) 2025 by Peixuan Song.
:license: MIT License, see LICENSE for more details.
"""

__version__ = "0.1.0"

import torch,gc
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

import utils.network as nt
from utils.trainer import Trainer
from utils.dataloader import CustomDataset

gc.collect()
torch.cuda.empty_cache()

retrain = 0

# hyperparameters
lr = 5e-4
epochs = 500
batch_size = 8
weight_decay = 1e-4
# the lr changing
stepsize = 30
gamma = 0.5

# create model & citerion
model = nt.ResAttU_Net(img_ch=1, output_ch=2) # recommended
nt.init_weights(model, 'kaiming')
criterion = nn.BCEWithLogitsLoss()

# create SummaryWriter
logdir = "log/" + 'net:ResAttU ' + 'retrain:01 ' + 'loss:bce '+ 'decay:(5e-4,15,0.5) '+ 'bs:64 '
writer = SummaryWriter(logdir)

# transform
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((1024, 1024)),
    transforms.ToTensor(),
])

# data set
data_dir = 'dataset'
dataset = CustomDataset(data_dir, transform=transform)
print(f'Loaded {len(dataset)} samples')
# divide dataset into training and testing sets
train_size = int(0.9 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# set device & parallel
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_ids = list(range(torch.cuda.device_count()))
model = nn.DataParallel(model, device_ids=device_ids).to(device)
print(f'Using devices: {device_ids}')

# optimizer and criterion
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)

# if fine-tuning
if retrain:
    model = torch.load('model.pth')
else:
    model.to(device)

# create trainer and train the model
print(f'Learning rate: {lr}; Batch size: {batch_size}; Epochs: {epochs}')
trainer = Trainer(model, optimizer, criterion, device, writer)
trainer.train(train_loader, test_loader, epochs=epochs)

# save the model
torch.save(model, 'model.pth')

# close the SummaryWriter
writer.close()
