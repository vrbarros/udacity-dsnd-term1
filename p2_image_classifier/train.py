import argparse
import torch
import time

from collections import OrderedDict
from os.path import isdir
from torch import nn
from torch import optim

from utils import get_data, get_category_names, get_device, get_architecture, set_classifier, save_checkpoint, load_checkpoint

def arg_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--save_dir', default="./checkpoint.pth", type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--hidden_units', default=200, type=int)
    parser.add_argument('--epochs', default=6, type=int)
    parser.add_argument('--gpu', default="gpu", type=str)
    parser.add_argument('--arch', default="vgg16", type=str)
    parser.add_argument('--debug', default=0, type=int)
    parser.add_argument('--resume', default=0, type=int)
    parser.add_argument('--category_names', default='cat_to_name.json')
    
    args = parser.parse_args()
    
    return args

def test(model, dataloaders, device):
    start = time.time()
        
    accuracy = 0
    
    model.to(device);
    model.eval()
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logps = model.forward(inputs)
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        print(
            f"Validation accuracy: {(accuracy / len(dataloaders['test'])) * 100:.3f}%.. "
            f"Time: {(time.time() - start)/3:.3f} seconds"
        )
    
    model.train()
    
    return (accuracy / len(dataloaders['valid'])) * 100

def train(model = None, epochs = 1, print_every = 5, debug = True, dataloaders = None, device = None, criterion = None, optimizer = None):
    start = time.time()
    
    steps = 0
    
    print("Training {} epochs and reporting every {} times, using {}... DEBUG={}".format(epochs, print_every, device, debug))

    for epoch in range(epochs):
        model.train()
        
        running_loss = 0

        for ii, (inputs, labels) in enumerate(dataloaders['train']):
            steps += 1

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()

            # Take an update step and few the new weights
            optimizer.step()

            running_loss += loss.item()

            if debug == 1 and ii == print_every * 2:
                print("Stop for debug!")
                break

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0

                model.eval()
                
                with torch.no_grad():
                    for valid_inputs, valid_labels in dataloaders['valid']:
                        valid_inputs, valid_labels = valid_inputs.to(device), valid_labels.to(device)

                        valid_logps = model.forward(valid_inputs)
                        batch_loss = criterion(valid_logps, valid_labels)
                        valid_loss += batch_loss.item()

                        ps = torch.exp(valid_logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == valid_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(
                    f"Epoch {epoch+1}/{epochs}.. "
                    f"Step {steps}.. "
                    f"Train loss: {running_loss / print_every:.3f}.. "
                    f"Test loss: {valid_loss / len(dataloaders['valid']):.3f}.. "
                    f"Test accuracy: {(accuracy / len(dataloaders['valid'])) * 100:.3f}%.. "
                    f"Time: {(time.time() - start) / 3:.3f} seconds"
                )

                running_loss = 0

                model.train()

def main():

    args = arg_parser()
    
    dataloaders, image_datasets = get_data()
    cat_to_name = get_category_names(args.category_names)
    device = get_device(args.gpu)
    
    if args.resume == 1:
        print("Resume training from {}".format(args.save_dir))
        
        model, params = load_checkpoint(args.save_dir)
        
        learn_rate = params['learn_rate']
        optimizer_state_dict = params['optimizer_state_dict']
        
        test_accuracy = test(model, dataloaders, device)
        
    else:
        print("Starting new training that will be saved to {}".format(args.save_dir))
        
        model = get_architecture(args.arch)
        model.classifier = set_classifier(model, args.hidden_units, args.arch)
        
        learn_rate = args.learning_rate
        
        
    model.to(device);
    
    optimizer = optim.Adam(params=model.classifier.parameters(), lr=learn_rate)
    
    if args.resume == 1:
        optimizer.load_state_dict(optimizer_state_dict)
    
    criterion = nn.NLLLoss()
    
    train(model, args.epochs, 40, args.debug, dataloaders, device, criterion, optimizer)
    
    test_accuracy = test(model, dataloaders, device)
    
    if args.debug == 0 and valid_accuracy < 70:
        raise Exception("Validation accuracy must be more than 70%")
        
    save_checkpoint(model, image_datasets, optimizer, learn_rate, args.hidden_units, args.save_dir, args.arch)
    
if __name__ == '__main__': main()