import os
import cv2
import numpy as np


def read_Dataset(path_dir):
    in_dir = os.listdir(path_dir)
    Images = []
    Target = []
    for i in range(len(in_dir)):
        file_dir = path_dir + '/' + in_dir[i]
        file = os.listdir(file_dir)
        for j in range(len(file)):  # len(file)
            print(i, j)
            file_name = file_dir + '/' + file[j]
            split = file_dir.split('_')
            if split[len(split) - 1] == 'mask.png':
                data = cv2.imread(file_dir)
                Images.append(data)
                Target.append(i)
    return Images, Target


# Read Dataset 1
an = 0
if an == 1:
    out_dir = './Data/data'
    data, Target = read_Dataset(out_dir)
    np.save('Data_1.npy', data)


def read_Dataset(path_dir):
    in_dir = os.listdir(path_dir)
    Images = []
    for i in range(len(in_dir)):
        file_dir = path_dir + '/' + in_dir[i]
        file = os.listdir(file_dir)
        for j in range(50):  # len(file)
            print(i, j)
            file_name = file_dir + '/' + file[j]
            data = cv2.imread(os.path.join(file_name))
            width = 256
            height = 256
            dim = (width, height)
            resized_image = cv2.resize(data, dim)
            Images.append(resized_image)
    return Images


# Read Dataset 2
an = 0
if an == 1:
    out_dir = './archive/Testing'
    data = read_Dataset(out_dir)
    np.save('Data_2.npy', data)

# Generate Ground_Truth
an = 0
if an == 1:
    Images = np.load('Data_2.npy', allow_pickle=True)
    GT = []
    for i in range(len(Images)):
        print(i)
        image = Images[i]
        img = np.zeros(image.shape, dtype=np.uint8)
        max_val = np.max(image)
        thresh = max_val - (max_val * 0.2)
        index = np.where(image >= thresh)
        img[index[0], index[1]] = 255
        img = img.astype(np.uint8)
        GT.append(img)
    np.save('GT_2.npy', GT)
