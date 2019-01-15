# Imports here
import matplotlib.pyplot as plt
import argparse
import torch
from torchvision import datasets, transforms
import torchvision.models as models
from torch import nn
from torch import optim
import seaborn as sns
import argparse
import json


def validation(model, validloader, criterion):
        test_loss = 0
        accuracy = 0
        for images, labels in validloader:

            images, labels = images.to('cuda:0'), labels.to('cuda:0')
            output = model.forward(images)
            test_loss += criterion(output, labels).item()

            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        
        return test_loss, accuracy


def train(data_dir, save_dir, arch, learning_rate, gpu, epochs, hidden_units):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'


    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], 
                                                               [0.229, 0.224, 0.225])])



    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)


    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, int(hidden_units))), 
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(int(hidden_units), 1000)),
                              ('relu', nn.ReLU()),
                              ('fc3', nn.Linear(1000, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(learning_rate))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if gpu:
        model.to('cuda')
    else:
        model.to('cpu')
    epochs = int(epochs)
    steps = 0
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        model.train()
        for images, labels in trainloader:
            steps += 1
            if gpu:
                images, labels = images.to('cuda:0'), labels.to('cuda:0')
            else:
                images = Variable(images.cpu(), volatile=True)
                labels = Variable(labels.cpu(), volatile=True)
            optimizer.zero_grad()

            output = model.forward(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(model, validloader, criterion)

                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))

                running_loss = 0

                # Make sure training is back on
                model.train()
    
    
    model.class_to_idx = train_data.class_to_idx

    checkpoint = {
                  'state_dict': model.state_dict(),
                  'model': model,
                 }

    torch.save(checkpoint, save_dir)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_directory", default="/home/aipnd-project/workspace/flowers",
                        type=str)
    parser.add_argument("--save_dir", default="/home/workspace/paind-project/checkpoint_new.pth",
                        type=str)
    parser.add_argument("--arch", help="choose your architecture", type=str, default="vgg16")
    parser.add_argument("--learning_rate", help="choose learning rate", type=str, default="0.001")
    parser.add_argument("--gpu", action='store_true', default=True)
    parser.add_argument("--epochs", help="choose number of epochs", type=str, default="1")
    parser.add_argument("--hidden_units", help="choose number of hidden_units", type=str, default="4096")
    args = parser.parse_args()
    train(args.data_directory, args.save_dir, args.arch, args.learning_rate, args.gpu, args.epochs, args.hidden_units)



