import torch
import json

from torch import nn
from collections import OrderedDict
from torchvision import datasets, transforms, models

THUMB_SIZE = 224
ORIG_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD_DEV = [0.229, 0.224, 0.225]
DEBUG = False

def get_data():
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomRotation(40),
                transforms.RandomResizedCrop(THUMB_SIZE),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD_DEV),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize(ORIG_SIZE),
                transforms.CenterCrop(THUMB_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD_DEV),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize(ORIG_SIZE),
                transforms.CenterCrop(THUMB_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD_DEV),
            ]
        ),
    }
    
    image_datasets = {
        "train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
        "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"]),
        "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
    }
    
    dataloaders = {
        "train": torch.utils.data.DataLoader(
            image_datasets["train"], batch_size=64, shuffle=True
        ),
        "valid": torch.utils.data.DataLoader(
            image_datasets["valid"], batch_size=64, shuffle=True
        ),
        "test": torch.utils.data.DataLoader(
            image_datasets["test"], batch_size=64, shuffle=True
        ),
    }
    
    return dataloaders, image_datasets
    
def get_category_names(file_path):
    with open(file_path, 'r') as file:
        cat_to_name = json.load(file)
        
    return cat_to_name


def get_device(args_gpu):
    if not args_gpu:
        return torch.device("cpu")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if (device is "cpu"):
        print("CUDA is not available. Using CPU instead.")
    
    return device

def get_architecture(args_arch):
    if args_arch == "vgg16":
        model = models.vgg16(pretrained=True)
        
    if args_arch == "alexnet":
        model = models.alexnet(pretrained=True)
        
    return model

def set_classifier(model, args_hidden_units = 1000, args_arch = None):
    """
    Create new untrained feed-forward network
    """
    
    drop_rate = 0.5
    
    for param in model.parameters():
        param.requires_grad = False
    
    if args_arch == "vgg16":
        classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(25088, 4096)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("drop1", nn.Dropout(p=drop_rate)),
                    ("fc2", nn.Linear(4096, 4096)),
                    ("relu2", nn.ReLU()),
                    ("fc3", nn.Linear(4096, args_hidden_units)),
                    ("relu3", nn.ReLU()),
                    ("fc4", nn.Linear(args_hidden_units, 102)),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )
    
    if args_arch == "alexnet":
        classifier = nn.Sequential(
            OrderedDict(
                [
                    ("fc1", nn.Linear(9216, 4096)),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("drop1", nn.Dropout(p=drop_rate)),
                    ("fc2", nn.Linear(4096, 4096)),
                    ("relu2", nn.ReLU()),
                    ("drop2", nn.Dropout(p=drop_rate)),
                    ("fc3", nn.Linear(4096, args_hidden_units)),
                    ("relu3", nn.ReLU()),
                    ("fc4", nn.Linear(args_hidden_units, 102)),
                    ("output", nn.LogSoftmax(dim=1)),
                ]
            )
        )
    
    return classifier

def save_checkpoint(model, image_datasets, optimizer, learn_rate, args_hidden_units, args_save_dir, args_arch):
    checkpoint = {
        'arch': args_arch,
        'model_state_dict': model.state_dict(),
        'class_to_idx': image_datasets['train'].class_to_idx,
        'optimizer_state_dict': optimizer.state_dict(),
        'learn_rate': learn_rate,
        'hidden_units': args_hidden_units,
    }

    torch.save(checkpoint, args_save_dir)
    
    print("Saved to {} with success!".format(args_save_dir))
    
    
def load_checkpoint(args_load_dir):
    checkpoint = torch.load(args_load_dir)
    
    arch = checkpoint['arch']
    
    model = get_architecture(arch)
    
    model.classifier = set_classifier(model, checkpoint['hidden_units'], arch)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    params = {
        'optimizer_state_dict': checkpoint['optimizer_state_dict'],
        'learn_rate': checkpoint['learn_rate'],
    }
    
    print("Loaded network with success from {}".format(args_load_dir))
    
    return model, params