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
from torch.autograd import Variable
from PIL import Image
import numpy as np

def label_mapping(path):
    with open(path, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def load_checkpoint(path):
    checkpoint = torch.load(path)
    model = checkpoint['model']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    return model

def process_image(image):
    normalize = transforms.Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])
    img_tensor = preprocess(image)
    np_image = np.array(img_tensor) 
            
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    
    image = image.transpose((1, 2, 0))
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk, cat_to_name, gpu):
    img = Image.open(image_path)
    img = process_image(img)
    if gpu:
        img = torch.from_numpy(img).type(torch.cuda.FloatTensor)
    img = img.unsqueeze(0) 
    with torch.no_grad():
        output = model.forward(img)
    ps = torch.exp(output)
    top_probs, top_labs = ps.topk(5)
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    if gpu:
        top_probs = top_probs.cuda().cpu().detach().numpy().tolist()[0]
        top_labs = top_labs.cuda().cpu().detach().numpy().tolist()[0]
    else:
        top_probs = top_probs.cpu().detach().numpy().tolist()[0]
        top_labs = top_labs.cpu().detach().numpy().tolist()[0]
    top_flowers = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    return top_probs, top_labs, top_flowers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", default="/home/workspace/aipnd-project/flowers/test/1/image_06743.jpg",
                        type=str)
    parser.add_argument("checkpoint_path", default="/home/workspace/paind-project/checkpoint_new.pth",
                        type=str)
    parser.add_argument("--topk", type=str, default="5")
    parser.add_argument("--category_names", type=str, default="/home/workspace/aipnd-project/cat_to_name.json")
    parser.add_argument("--gpu", action='store_true', default=True)
    args = parser.parse_args()
    cat_to_name = label_mapping(args.category_names)
    model = load_checkpoint(args.checkpoint_path)
    top_probs, top_classes, top_flowers = predict(args.image_path, model, int(args.topk), cat_to_name, args.gpu)
    print("Probabilities: ", top_probs)
    print("Top k classes: ", top_classes)
    print("Top Flowers: ", top_flowers)
