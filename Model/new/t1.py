import os

import torch

from Code.my_addition.Model import solver


# from data_loader import get_loader

# Print arguments


# Update arguments
def update_args(args):
    args.model_path = os.path.join(args.model_path, args.dataset)
    args.is_cuda = torch.cuda.is_available()  # Check GPU availability
    return args


def print_args(args):
    for k in dict(sorted(vars(args).items())).items():
        print(k)
    print()
    return


def predictions(img, rows, cols):
    prediction = solver.test_image(img, rows, cols)
    return prediction

# 假设图像路径
# image_path = '1.tif'
# img = cv2.imread(image_path, cv2.IMREAD_COLOR)
# points_test = np.load('points_test.npy')
# dir_test = np.load('directions_test.npy')
# rows= points_test[:,0]
# cols = points_test[:,1]
# # 进行预测
# prediction = predictions(img, rows, cols)
# print(prediction)