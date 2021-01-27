import argparse
import json
import torch
import numpy as np
import seaborn as sns

from PIL import Image
from math import ceil
from torchvision import models
from matplotlib import pyplot as plt

from utils import MEAN, STD_DEV, get_category_names, get_device, load_checkpoint

def arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--image', default="./flowers/test/10/image_07090.jpg", type=str)
    parser.add_argument('--checkpoint', default="./checkpoint.pth", type=str)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--gpu', default="gpu")
    parser.add_argument('--category_names', default='cat_to_name.json')

    args = parser.parse_args()
    
    return args

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model   
    image_size = (256, 256)
    crop_size = 244
    
    mean = np.array(MEAN)
    std = np.array(STD_DEV)
    
    image = Image.open(image)
    image = image.resize(image_size)
    
    height, width = image_size
    dimensions = {
        "left": (width - crop_size) / 2,
        "lower": (height - crop_size) / 2,
        "right": ((width - crop_size) / 2) + crop_size,
        "top": ((height - crop_size) / 2) + crop_size,
    }
    
    cropped_image = image.crop(tuple(dimensions.values()))
    
    assert cropped_image.size == (
        crop_size,
        crop_size,
    ), f"Image size {cropped_image.size}, must be {crop_size}"
        
    np_image = np.array(cropped_image)
    np_image = np_image / [255,255,255]
    np_image = (np_image - mean) / std
    np_image = np_image.transpose((2, 0, 1))

    return np_image

def predict(image_path, model_checkpoint, topk = 5, device = None, cat_to_name = None):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    with torch.no_grad():
        print("Image {} is being processed".format(image_path))
        sample = torch.from_numpy(np.expand_dims(process_image(image_path), axis=0)).type(torch.FloatTensor).to(device)
        
        model, params = load_checkpoint(model_checkpoint)
        model.to(device)
        model.eval();
        
        logps = model.forward(sample)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk)
        
        top_p = np.array(top_p.detach())[0]
        top_class = np.array(top_class.detach())[0]
        
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        
        top_class = [idx_to_class[key] for key in top_class]
        top_flowers = [cat_to_name[key] for key in top_class]
        
        print("top_p={} top_class={} top_flowers={}".format(top_p, top_class, top_flowers))
        
        return top_p, top_class, top_flowers

def main():
    
    args = arg_parser()
    
    cat_to_name = get_category_names(args.category_names)
    device = get_device(args.gpu)
    
    top_p, top_class, top_flowers = predict(args.image, args.checkpoint, args.top_k, device, cat_to_name)
    
if __name__ == '__main__': main()