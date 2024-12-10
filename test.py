import math
import os
import random
import shutil
import time
import warnings

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import models
from ops.config import parser
from training.schedule import lr_setter
from training.train import train
from training.validate import validate
from utilis.meters import AverageMeter
from utilis.saving import save_checkpoint

best_acc1 = 0

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../RaffeModelTraining"))
from src.data import create_dataloader
from main_stablenet import Config
'''
class Config:
    def __init__(self, data_source, data_label, dataset_path, image_height, image_width, encoder, task, shuffle, batch_size, num_threads):
        self.data_source = data_source
        self.data_label = data_label
        self.dataset_path = dataset_path
        self.image_height = image_height
        self.image_width = image_width
        self.encoder = encoder
        self.task = task
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_threads=num_threads
'''

def main():
    args = parser.parse_args()
    args.classes_num = 2

    args.log_path = os.path.join(args.log_base, args.dataset, "test_log.txt")

    if not os.path.exists(os.path.dirname(args.log_path)):
        os.makedirs(os.path.dirname(args.log_path))

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    main_worker(ngpus_per_node, args)


def main_worker(ngpus_per_node, args):

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    model = models.__dict__[args.arch](args=args)
    num_ftrs = model.fc1.in_features
    model.fc1 = nn.Linear(num_ftrs, args.classes_num)
    
    print(args.checkpoint_path)
    print("=> loading checkpoint '{}'".format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path, map_location=torch.device('cpu'))
    model.cuda(args.gpu)
    model.load_state_dict(checkpoint['state_dict'])

    cfg = Config(
        data_source="folder",
        data_label="test",
        dataset_path=args.data,
        image_height=256,
        image_width=256,
        encoder="imagenet",
        task="classification",
        shuffle=False,
        batch_size=128,
        num_threads=1,
        isTrain=False,
        no_crop=False,
        no_flip=True,
        augmentations=False,
        no_resize=True,
        blur_prob=0.5,
        blur_sig=[0.0, 3.0],
        jpg_prob=0.5,
        jpg_method=["cv2", "pil"],
        jpg_qual=[30, 100],
        cropSize=224,
    )
    
    cfg.data_label = "test"
    cfg.shuffle = False
    # cfg.batch_size = 1
    test_loader = create_dataloader(cfg)

    validate(test_loader, model, criterion, 0, True, args, tensor_writer=None)

if __name__ == '__main__':
    main()
