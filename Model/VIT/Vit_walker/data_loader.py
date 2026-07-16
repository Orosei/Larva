import os

import torch
from torchvision import datasets, transforms

from .five_loader import MyFiveLikeDataset, DataLoader


def get_loader(args):
    datapath = os.path.join(args.data_path, args.dataset)
    if args.dataset == 'fives':
        loader = DataLoader(datapath, 'Original', 'Ground truth')
        sample_num = 500
        train_num = int(loader.getnum() * 0.95)
        test_num = loader.getnum()
        train1 = loader.load_all_data(0, train_num, sample_num, args.image_size)
        test1 = loader.load_all_data(train_num, test_num, sample_num, args.image_size)

        net_size = [args.image_size, args.image_size]
        mean = [0.5, 0.5, 0.5]
        std = [0.25, 0.25, 0.25]
        train_transform = transforms.Compose([transforms.Resize(net_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean, std)])
        train = MyFiveLikeDataset(train1, train_transform, net_size)
        test = MyFiveLikeDataset(test1, train_transform, net_size)

    elif args.dataset == 'mnist':
        # Transforms for train
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                              transforms.RandomCrop(args.image_size, padding=2),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])])
        train = datasets.MNIST(datapath, train=True, download=True, transform=train_transform)

        # Transforms for test
        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])
        test = datasets.MNIST(datapath, train=False, download=True, transform=test_transform)
    elif args.dataset == 'fashionmnist':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                              transforms.RandomCrop(args.image_size, padding=2),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5], [0.5])])
        train = datasets.FashionMNIST(datapath, train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5], [0.5])])
        test = datasets.FashionMNIST(datapath, train=False, download=True, transform=test_transform)
    elif args.dataset == 'svhn':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                              transforms.RandomCrop(args.image_size, padding=2),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.4376821, 0.4437697, 0.47280442],
                                                                   [0.19803012, 0.20101562, 0.19703614])])
        train = datasets.SVHN(datapath, split='train', download=True, transform=train_transform)
        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.4376821, 0.4437697, 0.47280442],
                                                                  [0.19803012, 0.20101562, 0.19703614])])
        test = datasets.SVHN(datapath, split='test', download=True, transform=test_transform)
    elif args.dataset == 'cifar10':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                              transforms.RandomCrop(args.image_size, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandAugment(),
                                              # RandAugment augmentation for strong regularization
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        train = datasets.CIFAR10(datapath, train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])])
        test = datasets.CIFAR10(datapath, train=False, download=True, transform=test_transform)
    elif args.dataset == 'cifar100':
        train_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                              transforms.RandomCrop(args.image_size, padding=4),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandAugment(),
                                              # RandAugment augmentation for strong regularization
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
        train = datasets.CIFAR100(datapath, train=True, download=True, transform=train_transform)

        test_transform = transforms.Compose([transforms.Resize([args.image_size, args.image_size]),
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761])])
        test = datasets.CIFAR100(datapath, train=False, download=True, transform=test_transform)
    else:
        print("Unknown dataset")
        exit(0)

    # Define dataloaders
    train_loader = torch.utils.data.DataLoader(dataset=train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.n_workers,
                                               drop_last=True)

    test_loader = torch.utils.data.DataLoader(dataset=test,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.n_workers,
                                              drop_last=False)

    return train_loader, test_loader
