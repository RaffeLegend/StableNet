import torch
import numpy as np
import os
from PIL import Image
from torchvision import transforms

import torch.nn.functional as F
import matplotlib.pyplot as plt

import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import models
from ops.config import parser
from training.focal_frequency_loss import FocalFrequencyLoss

def load_model(args):
    model = models.__dict__[args.arch](args=args)
    num_ftrs = model.fc1.in_features
    model.fc1 = nn.Linear(num_ftrs, args.classes_num)

    print(args.checkpoint_path)
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.cuda(args.gpu)
    model.load_state_dict(checkpoint['state_dict'])
    return model

def compute_saliency_map(model, input_tensor, args, target_class):
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    focal_loss = FocalFrequencyLoss().cuda(args.gpu)
    input_tensor.requires_grad_()
    output, cfeatures, recon = model(input_tensor)
    target_class = torch.tensor([target_class]).cuda()
    loss1 = criterion(output, target_class)
    input = transforms.Resize((recon.shape[-2:]))(input_tensor)
    loss2 = focal_loss(recon, input)
    loss = loss1 + loss2
    model.zero_grad()
    loss2.backward()
    saliency, _ = torch.max(input_tensor.grad.data.abs(), dim=1)
    return saliency

def overlay_saliency_on_image(image_path, saliency_map):
    image = Image.open(image_path).convert('RGB')
    image = image.resize((saliency_map.shape[1], saliency_map.shape[0]))
    plt.imshow(image)
    plt.imshow(saliency_map, cmap=plt.cm.hot, alpha=0.5)
    plt.colorbar()
    plt.title('Saliency Map Overlay')
    plt.axis('off')
    plt.show()

def process_image(image_path):
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def generate_saliency_maps_for_folder(model, folder_path, args, target_class, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(folder_path) 
    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(folder_path, filename)
            input_tensor = process_image(image_path)
            input_tensor = input_tensor.cuda()
            saliency_map = compute_saliency_map(model, input_tensor, args, target_class)
            output_path = os.path.join(output_folder, f'saliency_{filename}')
            image = Image.open(image_path).convert('RGB')
            image = image.resize((saliency_map.shape[1], saliency_map.shape[0]))
            plt.imshow(image)
            plt.imshow(saliency_map, cmap=plt.cm.hot, alpha=0.5)
            plt.axis('off')
            # plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            # plt.close()
            plt.imsave(output_path, saliency_map[0].cpu().numpy(), cmap=plt.cm.hot)
            plt.close()

# Example usage:
model_path = "/mnt/data2/users/hilight/yiwei/train/checkpoints/paper/DomainSet/model_best.pth.tar"  # Your pre-trained model
folder_path = '/mnt/data2/users/hilight/datasets/ForenSynths/test/progan/bird/1_fake'
target_class = 1  # The target class index
output_folder = './result'

args = parser.parse_args()
args.classes_num = 2
model = load_model(args)
generate_saliency_maps_for_folder(model, folder_path, args, target_class, output_folder)
