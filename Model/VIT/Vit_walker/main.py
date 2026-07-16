import argparse
import datetime
import os

import cv2
import numpy as np
import torch

from .five_loader import select_white_pixels, save2pic
from .solver import Solver


def vit_int(train):
    args = init_model()
    solver = Solver(args, train)
    return solver


def vit_main(solver, train):
    if train:
        os.makedirs(solver.args.model_path, exist_ok=True)
        os.makedirs(solver.args.output_path, exist_ok=True)
        solver.train()
        solver.test()
    else:
        image = cv2.imread('.\\data\\data_200\imgorg\\subimg6.bmp')
        label = cv2.imread('.\\data\\data_200\img\\subimg6.bmp', cv2.IMREAD_GRAYSCALE)

        size = (512, 512)
        if image.shape[0] > size[0] or image.shape[1] > size[1]:
            image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR_EXACT)
            label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)

        coords = select_white_pixels(label, 50, 10)
        rows = [c[0] for c in coords]
        cols = [c[1] for c in coords]
        probs = solver.test_image(image, rows, cols)
        ids = np.argmax(probs, axis=0) * 45
        save2pic("test", f"id", image, label, image, label, ids, list(coords))
    return


# Print arguments
def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()
    return


# Update arguments
def update_args(args):
    args.model_path = os.path.join(args.model_path, args.dataset)
    args.output_path = os.path.join(args.output_path, args.dataset)
    args.n_patches = (args.image_size // args.patch_size) ** 2
    args.is_cuda = torch.cuda.is_available()  # Check GPU availability

    print("Using GPU") if args.is_cuda else print("Cuda not available.")
    print_args(args)
    return args


def init_model():
    parser = argparse.ArgumentParser(description='Simple PyTorch of Vision Transformer (ViT) from scratch')
    # Training Arguments
    parser.add_argument('--epochs', type=int, default=25, help='number of training epochs')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='number of epochs to warmup learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--n_classes', type=int, default=10, help='number of classes in the dataset')
    parser.add_argument('--n_workers', type=int, default=4, help='number of workers for data loaders')
    parser.add_argument('--lr', type=float, default=5e-4, help='peak learning rate')
    parser.add_argument('--output_path', type=str, default='./outputs', help='path to store plots')
    # Data arguments
    parser.add_argument('--dataset', type=str, default='fives',
                        choices=['fives', 'mnist', 'fashionmnist', 'svhn', 'cifar10', 'cifar100'], help='dataset')
    parser.add_argument("--image_size", type=int, default=64, help='images size')
    parser.add_argument("--patch_size", type=int, default=4, help='patch Size')
    parser.add_argument("--n_channels", type=int, default=3, help='number of channels')
    parser.add_argument('--data_path', type=str, default='./data/', help='path to store downloaded dataset')
    # ViT Arguments
    parser.add_argument("--embed_dim", type=int, default=128, help='dimensionality of the latent space')
    parser.add_argument("--n_attention_heads", type=int, default=4, help='Multi-head attention')
    parser.add_argument("--forward_mul", type=int, default=2, help='forward multiplier')
    parser.add_argument("--n_layers", type=int, default=8, help='number of encoder layers')
    parser.add_argument("--dropout", type=float, default=0.1, help='dropout value')
    parser.add_argument('--model_path', type=str, default='C:/Master/Snake/Snake_v1_2024_12_20/Code/Model/VIT/Vit_walker/model', help='path to store trained model')
    parser.add_argument("--load_model", type=bool, default=True, help="load saved model")
    args = parser.parse_args()
    args = update_args(args)
    return args


if __name__ == '__main__':
    train_mode = False
    start_time = datetime.datetime.now()
    print("Started at " + str(start_time.strftime('%Y-%m-%d %H:%M:%S')))
    solver = vit_int(train_mode)
    vit_main(solver, train_mode)

    end_time = datetime.datetime.now()
    duration = end_time - start_time
    print("Ended at " + str(end_time.strftime('%Y-%m-%d %H:%M:%S')))
    print("Duration: " + str(duration))
