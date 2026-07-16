import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from torchvision import transforms

from .data_loader import get_loader
from .model import VisionTransformer


class Solver(object):
    def __init__(self, args, train=True):
        self.args = args
        # Create object of the Vision Transformer
        self.model = VisionTransformer(n_channels=self.args.n_channels, embed_dim=self.args.embed_dim,
                                       n_layers=self.args.n_layers, n_attention_heads=self.args.n_attention_heads,
                                       forward_mul=self.args.forward_mul, image_size=self.args.image_size,
                                       patch_size=self.args.patch_size, n_classes=self.args.n_classes,
                                       dropout=self.args.dropout)
        if self.args.is_cuda:  # Push to GPU
            self.model = self.model.cuda()
        print('--------Network--------')
        print(self.model)

        # Option to load pretrained model
        if self.args.load_model:
            print("Using pretrained model")
            self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'ViT_model.pt')))

        if train:
            self.init_solver1()
        return

    def init_solver1(self):
        self.train_loader, self.test_loader = get_loader(self.args)
        self.loss_fn = nn.CrossEntropyLoss()
        self.train_losses = []
        self.train_accuracies = []

    def train(self):
        iters = len(self.train_loader)
        optimizer = optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=1e-3)
        # scheduler for linear warmup of lr and then cosine decay to 1e-5
        linear_warmup = LinearLR(optimizer, start_factor=1 / self.args.warmup_epochs, end_factor=1.0,
                                 total_iters=self.args.warmup_epochs - 1, last_epoch=-1, verbose=True)
        cos_decay = CosineAnnealingLR(optimizer=optimizer, T_max=self.args.epochs - self.args.warmup_epochs,
                                      eta_min=1e-5, verbose=True)

        for epoch in range(self.args.epochs):
            self.model.train()
            train_epoch_loss = []
            train_epoch_acc = []
            for i, (x, y) in enumerate(self.train_loader):
                if self.args.is_cuda:
                    x, y = x.cuda(), y.cuda()
                logits = self.model(x)
                loss = self.loss_fn(logits, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # Batch metrics
                batch_pred = logits.max(1)[1]
                acc = (y == batch_pred).float().mean()
                train_epoch_loss += [loss.item()]
                train_epoch_acc += [acc.item()]
                # Log training progress
                if i % 50 == 0 or i == (iters - 1):
                    print(f'Ep:{epoch + 1}/{self.args.epochs}\tIt:{i + 1}/{iters}\tloss:{loss:.4f}\tacc:{acc:.2%}')

            # Update training progression metric arrays
            linear_warmup.step() if epoch < self.args.warmup_epochs else cos_decay.step()
            torch.save(self.model.state_dict(), os.path.join(self.args.model_path, "ViT_model.pt"))
            self.train_losses += [sum(train_epoch_loss) / (iters + 1e-6)]
            self.train_accuracies += [sum(train_epoch_acc) / (iters + 1e-6)]
        self.plot_graphs()
        return

    def plot_graphs(self):
        plt.plot(self.train_losses, color='b', label='Train')

        plt.ylabel('Loss', fontsize=18)
        plt.yticks(fontsize=16)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)

        # plt.show()  # Uncomment to display graph
        plt.savefig(os.path.join(self.args.output_path, 'graph_loss.png'), bbox_inches='tight')
        plt.close('all')

        # Plot graph of accuracies
        plt.plot(self.train_accuracies, color='b', label='Train')
        plt.ylabel('Accuracy', fontsize=18)
        plt.yticks(fontsize=16)
        plt.xlabel('Epoch', fontsize=18)
        plt.xticks(fontsize=16)
        plt.legend(fontsize=15, frameon=False)

        # plt.show()  # Uncomment to display graph
        plt.savefig(os.path.join(self.args.output_path, 'graph_accuracy.png'), bbox_inches='tight')
        plt.close('all')
        return

    def test_dataset(self, loader):
        from sklearn.metrics import confusion_matrix, accuracy_score

        self.model.eval()

        all_labels = []
        all_logits = []
        for (x, y) in loader:
            if self.args.is_cuda:
                x = x.cuda()
            with torch.no_grad():  # Avoid capturing gradients in evaluation time for faster speed
                logits = self.model(x)
            all_labels.append(y)
            all_logits.append(logits.cpu())

        all_labels = torch.cat(all_labels)
        all_logits = torch.cat(all_logits)
        all_pred = all_logits.max(1)[1]
        loss = self.loss_fn(all_logits, all_labels).item()
        acc = accuracy_score(y_true=all_labels, y_pred=all_pred)
        cm = confusion_matrix(y_true=all_labels, y_pred=all_pred, labels=range(self.args.n_classes))
        return acc, cm, loss

    def test(self, train=False):
        if train:
            acc, cm, loss = self.test_dataset(self.train_loader)
        else:
            acc, cm, loss = self.test_dataset(self.test_loader)
        print(f"Test acc: {acc:.2%}\t, loss: {loss:.4f}\nConfusion Matrix:\n{cm}")
        return acc, loss

    def test_sub_image(self, batch_images):
        self.model.eval()
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
        batch_image_tensors = torch.stack([transform(Image.fromarray(img)) for img in batch_images]).cuda()
        with torch.no_grad():  # Avoid capturing gradients in evaluation time for faster speed
            model_output = self.model(batch_image_tensors)
            numpy_output = model_output.detach().cpu().numpy()
            max_indexes = np.argmax(numpy_output, axis=1)
        return max_indexes

    def test_image(self, img, rows, cols, starttime):
        self.model.eval()

        h, w = self.args.image_size, self.args.image_size
        batch_images = [crop_with_center(img, row, col, w, h) for row, col in zip(rows, cols)]
        ids = self.test_sub_image(batch_images)
        probs = np.zeros((8, len(rows)))
        for i, row_idx in enumerate(ids):
            # if starttime < 30:
            probs[row_idx, i] = 0.3
            probs[(row_idx + 4) % 8, i] = 0.7
            # else:
            #     probs[row_idx, i] = 0.5
            #     probs[(row_idx + 4) % 8, i] = 0.5
        return probs


def crop_with_center(img, center_row, center_col, width, height):
    half_w = width // 2
    half_h = height // 2
    start_r = center_row - half_h
    start_c = center_col - half_w
    end_r = start_r + height
    end_c = start_c + width

    pad_top = max(0, -start_r)
    pad_bottom = max(0, end_r - img.shape[0])
    pad_left = max(0, -start_c)
    pad_right = max(0, end_c - img.shape[1])

    if any([pad_top, pad_bottom, pad_left, pad_right]):
        padded_img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    else:
        padded_img = img
    return padded_img[start_r + pad_top:end_r + pad_top, start_c + pad_left:end_c + pad_left]
