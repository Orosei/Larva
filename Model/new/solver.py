import os

import cv2
import torch
from torchvision import transforms
from PIL import Image

from Code.my_addition.Model import DirectionNet


# from drive_loader import indices_to_angles, draw_local_direction, show_image


class Test(object):
    def __init__(self, args):
        self.args = args
        # Get data loaders
        # self.train_loader, self.test_loader = get_loader(args)
        self.model = DirectionNet()
        if self.args.is_cuda:  # Push to GPU
            self.model = self.model.cuda()
        self.model.load_state_dict(torch.load(os.path.join(self.args.model_path, 'DirectionNet_best.pt')))

    def test_image(self, image, rows, cols):
        self.model.eval()
        h, w = self.args.image_size, self.args.image_size
        batch_images = [crop_with_center(image, row, col, w, h) for row, col in zip(rows, cols)]
        # show_image(batch_images)
        # print (batch_images.shape[2])
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5,0.5,0.5], [0.25,0.25,0.25])])
        batch_image_tensors = torch.stack([transform(Image.fromarray(img)) for img in batch_images]).cuda()
        with torch.no_grad():
            model_output = self.model(batch_image_tensors)
            _, indices = torch.topk(model_output, k=2, dim=1)
            indices = indices.detach().cpu().numpy()

        return indices

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