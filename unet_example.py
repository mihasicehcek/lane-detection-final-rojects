import torch
import torch.nn as nn
import cv2
import numpy as np
import torch.nn.functional as F
import time


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

model = torch.load('./models/lines_segm_v6_2datasets_ep19_128_128_3_full.pth', map_location=torch.device('cpu'))


model = model.to(torch.device('cpu'))

model.eval()

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

video = cv2.VideoCapture("4434242-uhd_2160_3840_24fps.mp4")


output_file = './video_unet.mp4'
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_size = ((frame_width//3)*2, frame_height//3)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_file, fourcc, 20, frame_size)


while True:
    ret, target = video.read()
    if not ret:
        print('End of video.')
        break

    originalWidth = target.shape[0]

    originalHeight = target.shape[1]

    mask_np = F.sigmoid(model(preprocessImage(target))).cpu().detach().numpy()[0].transpose(1, 2, 0)
    # mask_np = (mask_np > 0.3).astype(int)
    mask_np = mask2FullSize(mask_np, (originalHeight//3, originalWidth//3))

    mask_display = (mask_np).astype(np.uint8)

    mask_display = cv2.cvtColor(mask_display, cv2.COLOR_GRAY2BGR)

    target = cv2.resize(target, (originalHeight // 3, originalWidth // 3))
    combined_image = np.concatenate((target, mask_display), axis=1)

    cv2.imshow('Video Frame', combined_image)
    out.write(combined_image)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
end_time = time.time()

video.release()
out.release()
cv2.destroyAllWindows()
