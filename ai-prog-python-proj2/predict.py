import numpy as np
import torch
import torchvision
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch import nn
from torch import optim
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

parser = argparse.ArgumentParser(description='trains a machine learning model to classify images')

parser.add_argument('--model', type=str,
                    help='determines the pretrained network to use. ARGUMENT REQUIRED: options include vgg and densenet')
parser.add_argument('--epochs', type=int, help='determines number of epochs the NN will be trained for, default=1')
parser.add_argument('--learning_rate', type=float, help='determines learning rate of NN. default = 0.005')
parser.add_argument('--device', type=str,
                    help='determines which device the NN will be trained on. options are gpu and cpu, default=cpu')

args = parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

data_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train_data = torchvision.datasets.ImageFolder(root=train_dir, transform=train_transforms)
valid_data = torchvision.datasets.ImageFolder(root=valid_dir, transform=data_transforms)
test_data = torchvision.datasets.ImageFolder(root=test_dir, transform=data_transforms)

train_dataloader = DataLoader(train_data, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# print(args.model)
if args.model == 'vgg':
    model = torchvision.models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Sequential(nn.Linear(25088, 1000),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(1000, 500),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(500, 102),
                                     nn.LogSoftmax(dim=1))

elif args.model == 'resnet':
    model = torchvision.models.resnet18(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1000, 500),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(500, 102),
                                     nn.LogSoftmax(dim=1))
else:
    raise Exception('Please use a supported model. Supported models are vgg and resnet')

device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'gpu' else "cpu")
optimizer = optim.Adam(model.classifier.parameters(),
                       lr=args.learning_rate if args.learning_rate is not None else 0.005)
criterion = nn.NLLLoss()
model.to(device)

epochs = args.epochs if args.epochs is not None else 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in train_dataloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in valid_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)

                    valid_loss += batch_loss.item()

                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch + 1}/{epochs}.. "
                  f"Train loss: {running_loss / print_every:.3f}.. "
                  f"Validation loss: {valid_loss / len(valid_dataloader):.3f}.. "
                  f"Validation accuracy: {accuracy / len(valid_dataloader):.3f}")
            running_loss = 0
            model.train()

model.class_to_idx = train_data.class_to_idx
check_point = {'model': model.state_dict(),
               'index_vals': model.class_to_idx,
               'opti_state': optimizer.state_dict(),
               'model_to_use': args.model}
torch.save(check_point, 'checkpoint.pth')
