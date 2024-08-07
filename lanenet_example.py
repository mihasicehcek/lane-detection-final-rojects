import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
# import dataset_tools as dtools

import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np
import json
import random
import math

from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import KMeans, OPTICS
import time
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from skimage.morphology import skeletonize

from sklearn.metrics import silhouette_score

from scipy.interpolate import UnivariateSpline

from sklearn.linear_model import LinearRegression


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LaneNet(nn.Module):
    def __init__(self):
        super(LaneNet, self).__init__()
        self.encoder = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),  # размер будет 64x64
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),  # размер будет 32x32
            ConvBlock(128, 256),
            nn.MaxPool2d(2, 2),  # размер будет 16x16
            ConvBlock(256, 512),
            nn.MaxPool2d(2, 2)  # размер будет 16x16
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(256, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(128, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(64, 64),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            ConvBlock(32, 32)
        )

        self.embedding = nn.Conv2d(32, 3, kernel_size=1)

        self.output_conv = nn.Conv2d(32, 1, kernel_size=1)  # приводим к 1 каналу

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        embedding = self.embedding(x)
        x = self.output_conv(x)
        return x, embedding


model2 = torch.load('./models/lines_segm_lanet_v1_2datasets_ep16_128_128_3_full.pth', map_location=torch.device('cpu'))

model2 = model2.to(torch.device('cpu'))

model2.eval()


def preprocessImage(image):
    original_shape = image.shape[:2]

    half_height = original_shape[0] // 2
    image = image[half_height:, :, :]

    image = cv2.resize(image, (128, 128))
    image = image.transpose((2, 0, 1)) / 255.0

    return torch.tensor(image, dtype=torch.float32).unsqueeze(0)


def fullSizeMask(predictedMask):
    full_mask = np.zeros((predictedMask.shape[0] * 2, predictedMask.shape[1]))

    full_mask[predictedMask.shape[0]:, :] = predictedMask
    return full_mask


def mask2FullSize(mask, targeDim):
    mask_np = mask.squeeze() * 255
    mask_np = fullSizeMask(mask_np)
    mask_np = cv2.resize(mask_np, targeDim)
    return mask_np


def clusteredImage(res, embedding):
    H, W = embedding.shape[1], embedding.shape[2]
    binary_mask = (res > 0.2)

    x_coord, y_coord = np.meshgrid(np.arange(W), np.arange(H))
    x_coord_flat = x_coord.flatten()
    y_coord_flat = y_coord.flatten()

    embedding_flat = embedding.permute(1, 2, 0).reshape(-1, 3).detach().numpy()

    embedding_flat = np.column_stack((embedding_flat, x_coord_flat, y_coord_flat))

    binary_mask_flat = binary_mask.reshape(-1)

    embedding_flat = embedding_flat[binary_mask_flat]

    # break
    scaler = StandardScaler()
    embedding_flat = scaler.fit_transform(embedding_flat)

    scale_factor = 2
    embedding_flat[:, -2:] *= scale_factor

    dbscan = DBSCAN(eps=0.9, min_samples=10)
    clusters = dbscan.fit_predict(embedding_flat)

    clusters_image = -np.ones(binary_mask.shape)  # Инициализируем -1 для фоновых пикселей
    clusters_image[binary_mask] = clusters
    clusters_image[clusters_image == -1] = 0

    # poly
    unique_labels = np.unique(clusters)
    cluster_coords = {label: [] for label in unique_labels if label != -1 and label != 0}
    for i in range(clusters_image.shape[0]):
        for j in range(clusters_image.shape[1]):
            label = clusters_image[i, j]
            if label != -1 and label != 0:
                cluster_coords[label[0]].append((j, i))

    cmap = plt.get_cmap('tab20')
    unique_labels = np.unique(clusters_image)
    colors = cmap(np.linspace(0, 1, len(unique_labels)))

    cluster_colored_image = np.zeros((clusters_image.shape[0], clusters_image.shape[1], 3), dtype=np.uint8)
    # clusters_image = clusters_image[:, :, 0]
    # for label, color in zip(unique_labels, colors):
    #     if label != 0:  # Пропускаем фон
    #         color = (color[2] * 255, color[1] * 255, color[0] * 255)  # Преобразуем цвет из RGBA в BGR
    #         cluster_colored_image[clusters_image == label] = color

    clusters_image = cluster_colored_image

    poly_coeffs_list = []
    for label, color in zip(unique_labels, colors):
        if label != -1 and label != 0:
            coords = np.array(cluster_coords[label])
            x = coords[:, 0]
            y = coords[:, 1]

            try:
                # Вычисляем коэффициенты полинома второй степени
                poly_coeffs = np.polyfit(x, y, 2)
                poly = np.poly1d(poly_coeffs)

                # Генерируем новые x координаты для сглаживания линии
                x_new = np.linspace(x.min(), x.max(), 100)
                y_new = poly(x_new)

                # Преобразуем цвет для OpenCV
                color_cv = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

                # Рисуем кривую на изображении
                for i in range(1, len(x_new)):
                    pt1 = (int(x_new[i - 1]), int(y_new[i - 1]))
                    pt2 = (int(x_new[i]), int(y_new[i]))
                    cv2.line(clusters_image, pt1, pt2, color_cv, 1)
            except np.RankWarning:
                pass


    return clusters_image


# img = cv2.imread('./data/dashcam.jpg')
# originalWidth = img.shape[0]
#
# originalHeight = img.shape[1]
# res, embedding = model2(preprocessImage(img))
#
# mask_np = F.sigmoid(res).cpu().detach().numpy()[0].transpose(1, 2, 0)
# embedding = embedding.squeeze(0)
#
# clusters_image = clusteredImage(mask_np, embedding)
#
# empty_block = np.zeros((clusters_image.shape[0], clusters_image.shape[1], clusters_image.shape[2]),
#                        dtype=clusters_image.dtype)
#
#
# clusters_image = np.concatenate((empty_block, clusters_image), axis=0)
#
# clusters_image = cv2.resize(clusters_image, (originalHeight//3, originalWidth//3),
#                                     interpolation=cv2.INTER_NEAREST)
#
# embedding_image = embedding.detach().numpy().transpose(1, 2, 0)
# embedding_image = np.clip(embedding_image, 0, 1)
# embedding_image = (embedding_image*255).astype(np.uint8)
#
# embedding_image = np.concatenate((empty_block, embedding_image), axis=0)
#
# embedding_image = cv2.resize(embedding_image, (originalHeight//3, originalWidth//3),
#                                     interpolation=cv2.INTER_NEAREST)
#
# mask_np = mask2FullSize(mask_np, (originalHeight // 3, originalWidth // 3))
# mask_display = (mask_np).astype(np.uint8)
#
# mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)
#
# print(img)
# target = cv2.resize(img, (originalHeight // 3, originalWidth // 3))
# combined_image = np.concatenate((target, mask_display, embedding_image, clusters_image), axis=1)
#
# plt.figure(figsize=(16, 10))
# plt.imshow(combined_image)
# plt.show()

video = cv2.VideoCapture("4434242-uhd_2160_3840_24fps.mp4")


output_file = './video_lanenet.mp4'
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = ((frame_width//3)*4, frame_height//3)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, 20, frame_size)

while True:
    ret, target = video.read()
    if not ret:
        print('End of video.')
        break

    originalWidth = target.shape[0]

    originalHeight = target.shape[1]

    res, embedding = model2(preprocessImage(target))

    mask_np = F.sigmoid(res).cpu().detach().numpy()[0].transpose(1, 2, 0)
    embedding = embedding.squeeze(0)

    clusters_image = clusteredImage(mask_np, embedding)
    empty_block = np.zeros((clusters_image.shape[0], clusters_image.shape[1], clusters_image.shape[2]), dtype=clusters_image.dtype)
    clusters_image = np.concatenate((empty_block, clusters_image), axis=0)

    clusters_image = cv2.resize(clusters_image, (originalHeight//3, originalWidth//3),
                                        interpolation=cv2.INTER_NEAREST)

    embedding_image = embedding.detach().numpy().transpose(1, 2, 0)
    embedding_image = np.clip(embedding_image, 0, 1)
    embedding_image = (embedding_image*255).astype(np.uint8)

    embedding_image = np.concatenate((empty_block, embedding_image), axis=0)

    embedding_image = cv2.resize(embedding_image, (originalHeight//3, originalWidth//3),
                                        interpolation=cv2.INTER_NEAREST)

    mask_np = mask2FullSize(mask_np, (originalHeight // 3, originalWidth // 3))
    mask_display = (mask_np).astype(np.uint8)

    mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)


    target = cv2.resize(target, (originalHeight // 3, originalWidth // 3))
    combined_image = np.concatenate((target, mask_display, embedding_image, clusters_image), axis=1)

    cv2.imshow('Video Frame', combined_image)
    out.write(combined_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

video.release()
out.release()
cv2.destroyAllWindows()