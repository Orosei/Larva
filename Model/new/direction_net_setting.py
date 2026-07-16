import argparse
import os

import torch
from setting import root

from Code.my_addition.Model import Test


def update_args(args):
    args.model_path = os.path.join(args.model_path, args.dataset)
    args.is_cuda = torch.cuda.is_available()  # Check GPU availability
    return args


parser = argparse.ArgumentParser(description='Simple PyTorch of Vision Transformer (ViT) from scratch')
parser.add_argument('--dataset', type=str, default='drive_20',
                    choices=['fives', 'drive', 'drive_20', 'drive_test', 'mnist', 'fashionmnist', 'svhn', 'cifar10',
                             'cifar100'], help='dataset')
parser.add_argument("--image_size", type=int, default=32, help='images size')

parser.add_argument('--model_path', type=str, default=os.path.join(root, "Code\\Model\\new\\model"),
                    help='path to store trained model')
parser.add_argument("--load_model", type=bool, default=False, help="load saved model")
args = parser.parse_args()
args = update_args(args)
solver = Test(args)


