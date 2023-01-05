# U-Net 데이터 아키텍처 구현하기
# U-Net
# Biomedical 분양를 위해 만들어진 FCN 기반의 모델
# 오토인코더(autoencoder)와 디코더(decoder)기반 모델
# 인코딩 단계에서는 입력 이미지의 특징을 포착할 수 있도록 채널의 수를 늘리면서 차원을 축소
# 디코딩 단계에서는 저차원으로 인코딩된 정보만을 이용하여 채널의 수를 줄이고 차원을 늘려서 고차원의 이미지를 복원

# filter
# 특징이 데이터에 있는지 없는지 검출하는 함수
# 필터는 행렬로 정의
# 입력 받은 이미지 모두 행렬로 변환
# 입력 받은 데이터에서 그 특징을 가지고 있으면 결과값이 큰값이 나오며, 특징을 가지고 있지 않으면 0에 가까운 값이 반환
# 필터를 적용해서 얻어낸 결과를 feature map 또는 activation map이라고 부름

# stride
# 필터를 적용하는 간격
# 우측으로 한칸씩 이동, 아래로 한칸씩 이동하여 적용

# pooling
# feature map의 사이즈를 줄이는 방법
# max pooling을 가장 많이 사용
# 데이터의 크기를 줄이고 싶을 때 선택적으로 사용

# fc layter(fully connected layer)
# 기존의 뉴럴 네트워크

# dropout layer
# 오버피팅 방지를 위한 레이어
# 뉴럴 네트워크가 학습중일 때 랜덤하게 값을 발생하여 학습을 방해함으로 학습용 데이터에 결과가 치우치는 것을 방지함

import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.datasets as datasets
import torch.nn as nn
from project2 import tr_dataloader, build_dataloader, build_transformer
import cv2
import torch
from torchvision import transforms, models
from ipywidgets import interact
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
import copy
import pandas as pd

def ConvLayer(in_channels, out_channels):
    layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )
    return layers


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block1 = ConvLayer(in_channels=3, out_channels=64)
        self.conv_block2 = ConvLayer(in_channels=64, out_channels=128)
        self.conv_block3 = ConvLayer(in_channels=128, out_channels=256)
        self.conv_block4 = ConvLayer(in_channels=256, out_channels=512)
        self.conv_block5 = ConvLayer(in_channels=512, out_channels=1024)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        encode_features = []
        out = self.conv_block1(x)
        encode_features.append(out)
        out = self.pool(out)

        out = self.conv_block2(out)
        encode_features.append(out)
        out = self.pool(out)

        out = self.conv_block3(out)
        encode_features.append(out)
        out = self.pool(out)

        out = self.conv_block4(out)
        encode_features.append(out)
        out = self.pool(out)

        out = self.conv_block5(out)
        return out, encode_features

encoder = Encoder()
x = torch.randn(1, 3, 224, 224)
# print(x)
out, ftrs = encoder(x)
# print(out)
# print(ftrs)

for ftr in ftrs:
    print(ftr.shape)

print(out.shape)

# ConvTranspose2d: Deconvolution 작업을 수행
def UpConvLayer(in_channels, out_channels):
    layers = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return layers


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upconv_layer1 = UpConvLayer(in_channels=1024, out_channels=512)
        self.conv_block1 = ConvLayer(in_channels=512 + 512, out_channels=512)

        self.upconv_layer2 = UpConvLayer(in_channels=512, out_channels=256)
        self.conv_block2 = ConvLayer(in_channels=256 + 256, out_channels=256)

        self.upconv_layer3 = UpConvLayer(in_channels=256, out_channels=128)
        self.conv_block3 = ConvLayer(in_channels=128 + 128, out_channels=128)

        self.upconv_layer4 = UpConvLayer(in_channels=128, out_channels=64)
        self.conv_block4 = ConvLayer(in_channels=64 + 64, out_channels=64)

    def forward(self, x, encoder_features):
        out = self.upconv_layer1(x)
        croped_enc_feature = self._center_crop(encoder_features[-1], out.shape[2:])
        out = torch.cat([out, croped_enc_feature], dim=1)
        out = self.conv_block1(out)

        out = self.upconv_layer2(out)
        croped_enc_feature = self._center_crop(encoder_features[-2], out.shape[2:])
        out = torch.cat([out, croped_enc_feature], dim=1)
        out = self.conv_block2(out)

        out = self.upconv_layer3(out)
        croped_enc_feature = self._center_crop(encoder_features[-3], out.shape[2:])
        out = torch.cat([out, croped_enc_feature], dim=1)
        out = self.conv_block3(out)

        out = self.upconv_layer4(out)
        croped_enc_feature = self._center_crop(encoder_features[-4], out.shape[2:])
        out = torch.cat([out, croped_enc_feature], dim=1)
        out = self.conv_block4(out)
        return out

    def _center_crop(self, encoder_feature, decoder_feature_size):
        croped_features = transforms.CenterCrop(size=decoder_feature_size)(encoder_feature)
        return croped_features

encoder = Encoder()
decoder = Decoder()
x = torch.randn(1, 3, 224, 224)
out, ftrs = encoder(x)
out = decoder(out, ftrs)

print(out.shape)

import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, num_classes, retain_input_dim=True):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)
        self.retain_input_dim = retain_input_dim

    def forward(self, x):
        out, encode_features = self.encoder(x)
        out = self.decoder(out, encode_features)
        out = self.head(out)
        if self.retain_input_dim:
            _, _, H, W = x.shape
            out = F.interpolate(out, size=(H, W))
        return out

model = UNet(num_classes=4)
x = torch.randn(1, 3, 224, 224)
out = model(x)
print(f'Input shape: {x.shape}')
print(f'Output shape: {out.shape}')

# Dice similarity coefficient 설명 및 구현하기
# F1 Score와 개념상 같지만 영상처리에서 더 많이 사용함
# 2 * (정밀도 * 재현율) / (정밀도 + 재현율)
# 라벨링된 영역과 예측한 영역이 정확히 같다면 1이되며, 그렇지 않을 경우에는 0이 됨
# 영상 이미지등에서 정답과 예측값간의 차이를 알기위해 사용
# Segmentation에서 쓰이는 지표

for index, batch in enumerate(tr_dataloader):
    image = batch[0]
    targets = batch[1]
    predictions = model(image)

    if index == 0:
        break

num_classes = 4
predictions_ = torch.argmax(predictions, dim=1)
onehot_pred = F.one_hot(predictions_, num_classes=num_classes).permute(0, 3, 1, 2) # shape 변경
onehot_target = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2)

onehot_pred_ = onehot_pred[0]
onehot_target_ = onehot_target[0]

dice_coeff = 0
for class_index in range(1, num_classes):
    a = onehot_pred_[class_index]
    b = onehot_target_[class_index]
    set_inter = torch.dot(a.reshape(-1).float(), b.reshape(-1).float())
    set_sum = a.sum() + b.sum()
    dice_coeff += (2 * set_inter) / (set_sum + 1e-9)
dice_coeff /= (num_classes-1)

dice_coeff

dice_loss = 1. - dice_coeff

dice_loss


class UNet_metric():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def __call__(self, pred, target):
        onehot_pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=num_classes).permute(0, 3, 1, 2)
        onehot_target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
        dice_loss = self._get_dice_loss(onehot_pred, onehot_target)
        dice_coefficient = self._get_batch_dice_coefficient(onehot_pred, onehot_target)
        return dice_loss, dice_coefficient

    def _get_dice_coefficient(self, pred, target):
        set_inter = torch.dot(pred.reshape(-1).float(), target.reshape(-1).float())
        set_sum = pred.sum() + target.sum()
        if set_sum.item() == 0:
            set_sum = 2 * set_inter
        dice_coeff = (2 * set_inter) / (set_sum + 1e-9)
        return dice_coeff

    def _get_multiclass_dice_coefficient(self, pred, target):
        dice = 0
        for class_index in range(1, self.num_classes):
            dice += self._get_dice_coefficient(pred[class_index], target[class_index])
        return dice / (self.num_classes - 1)

    def _get_batch_dice_coefficient(self, pred, target):
        num_batch = pred.shape[0]
        dice = 0
        for batch_index in range(num_batch):
            dice += self._get_multiclass_dice_coefficient(pred[batch_index], target[batch_index])
            return dice / num_batch

    def _get_dice_loss(self, pred, target):
        return 1 - self._get_batch_dice_coefficient(pred, target)

criterion = UNet_metric(num_classes=4)
print(criterion(predictions, targets))


class UNet_metric():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.CE_loss = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, pred, target):
        loss1 = self.CE_loss(pred, target)
        onehot_pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=num_classes).permute(0, 3, 1, 2)
        onehot_target = F.one_hot(target, num_classes=num_classes).permute(0, 3, 1, 2)
        loss2 = self._get_dice_loss(onehot_pred, onehot_target)
        loss = loss1 + loss2
        dice_coefficient = self._get_batch_dice_coefficient(onehot_pred, onehot_target)
        return loss, dice_coefficient

    def _get_dice_coefficient(self, pred, target):
        set_inter = torch.dot(pred.reshape(-1).float(), target.reshape(-1).float())
        set_sum = pred.sum() + target.sum()
        if set_sum.item() == 0:
            set_sum = 2 * set_inter
        dice_coeff = (2 * set_inter) / (set_sum + 1e-9)
        return dice_coeff

    def _get_multiclass_dice_coefficient(self, pred, target):
        dice = 0
        for class_index in range(1, self.num_classes):
            dice += self._get_dice_coefficient(pred[class_index], target[class_index])
        return dice / (self.num_classes - 1)

    def _get_batch_dice_coefficient(self, pred, target):
        num_batch = pred.shape[0]
        dice = 0
        for batch_index in range(num_batch):
            dice += self._get_multiclass_dice_coefficient(pred[batch_index], target[batch_index])
            return dice / num_batch

    def _get_dice_loss(self, pred, target):
        return 1 - self._get_batch_dice_coefficient(pred, target)

criterion = UNet_metric(num_classes=4)
criterion(predictions, targets)

optimizer = torch.optim.SGD(model.parameters(), lr=1E-3, momentum=0.9)

optimizer.step()

for index, batch in enumerate(tr_dataloader):
    images = batch[0]
    targets = batch[1]
    predictions = model(images)

    if index == 0:
        break


def train_one_epoch(dataloaders, model, optimizer, criterion, device):
    losses = {}
    dice_coefficients = {}

    for phase in ['train', 'val']:
        running_loss = 0.0
        running_dice_coeff = 0.0

        if phase == 'train':
            model.train()
        else:
            model.eval()

        for index, batch in enumerate(dataloaders[phase]):
            images = batch[0].to(device)
            targets = batch[1].to(device)

            with torch.set_grad_enabled(phase == 'train'):
                predictions = model(images)
                loss, dice_coefficient = criterion(predictions, targets)

            if phase == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            running_loss += loss.item()
            running_dice_coeff += dice_coefficient.item()

            if phase == 'train':
                if index > 0 and index % 100 == 0:
                    text = f'{index}/{len(dataloaders[phase])}' + \
                           f' - Running Loss: {loss.item():.4f}' + \
                           f' - Running Dice: {dice_coefficient.item():.4f}'
                    print(text)
        losses[phase] = running_loss / len(dataloaders[phase])
        dice_coefficients[phase] = running_dice_coeff / len(dataloaders[phase])
    return losses, dice_coefficients

def save_model(model_state, model_name, save_dir='./trained_model'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model_state, os.path.join(save_dir, model_name))

data_dir = './Medical/'

NUM_CLASSES = 4
IMAGE_SIZE = 224
BATCH_SIZE = 12
DEVICE = 'cpu'


dataloaders = build_dataloader(data_dir, batch_size=BATCH_SIZE)
model = UNet(num_classes = NUM_CLASSES)
model = model.to(DEVICE)
criterion = UNet_metric(num_classes=NUM_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr=1E-3, momentum=0.9)

num_epochs = 10

best_epoch = 0
best_score = 0.0
train_loss, train_dice_coefficient = [], []
val_loss, val_dice_coefficient = [], []

for epoch in range(num_epochs):
    losses, dice_coefficients = train_one_epoch(dataloaders, model, optimizer, criterion, DEVICE)
    train_loss.append(losses['train'])
    val_loss.append(losses['val'])
    train_dice_ceofficient.append(dice_coefficients['train'])
    val_dice_ceofficient.append(dice_coefficients['val'])

    print(f"{epoch}/{num_epochs} - Train Loss: {losses['train']:.4f}, Val Loss: {losses['Val']: 4f}")
    print(f"{epoch}/{num_epochs} - Train Dice Coeff: {dice_coefficients['train']:.4f}, Val Dice Coeff: {dice_coefficients['Val']: 4f}")

    if (epoch > 3) and (dice_coefficients['val'] > best_score):
        best_epoch = epoch
        best_score = dice_coefficients['val']
        save_model(model.state_dict(), f'model_{epoch:02d}.pth')

print('Best epoch: {best_epoch} -> Best Dice Coeffient: {best_score:.4f}')

plt.figure(figsize=(6, 5))
plt.subplot(211)
plt.plot(train_loss, label="train")
plt.plot(val_loss,  label="val")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid("on")
plt.legend()
plt.subplot(212)
plt.plot(train_dice_coefficient, label="train")
plt.plot(val_dice_coefficient, label="val")
plt.xlabel("epoch")
plt.ylabel("dice coefficient")
plt.grid("on")
plt.legend()
plt.tight_layout()

# 모델 테스트 및 filtering 적용하기
def load_model(ckpt_path, num_classes, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = UNet(num_classes=num_classes)
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()
    return model

NUM_CLASSES = 4
IMAGE_SIZE = 224
# is_cuda = True
# DEVICE = torch.device('cuda' if torch.cuda.is_available and is_cuda else 'cpu')
DEVICE = 'cpu'

ckpt_path = './trained_model/model_09.pth'
model = load_model(ckpt_path, NUM_CLASSES, DEVICE)

transformer = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def morpholocal_process(mask, num_classes, ksize=7):
    new_mask = mask.copy()
    # cv2.getStructuringElement(): 커널의 구조 요소 생성. size에 None을 지정하면 기본값으로 3*3 사각형 요소를 생성
    # MORPH_RECT: 사각형, MORPH_CROSS: 십자가 모양, MORPH_ELLIPSE: 사각형에 내접하는 타원
    # anchor: MORPH_CROSS 모양의 구조 요소에서 고정점 좌표. (-1, -1)을 지정하면 구조 요소의 중앙을 고정점으로 사용
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))

    for class_index in range(1, num_classes):
        binary_mask = (mask == class_index).astype(np.uint8)
        closing = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        new_mask[closing.astype(np.bool_)] = class_index
    return new_mask

CLASS_ID_TO_RGB = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
    3: (0, 0, 255)
}


def decode_segmap(mask, num_classes):
    mask_H, mask_W = mask.shape
    R_channel = np.zeros((mask_H, mask_W), dtype=np.uint8)
    G_channel = np.zeros((mask_H, mask_W), dtype=np.uint8)
    B_channel = np.zeros((mask_H, mask_W), dtype=np.uint8)

    for class_index in range(1, num_classes):
        R_channel[mask == class_index] = CLASS_ID_TO_RGB[class_index][0]
        G_channel[mask == class_index] = CLASS_ID_TO_RGB[class_index][1]
        B_channel[mask == class_index] = CLASS_ID_TO_RGB[class_index][2]

    RGB_mask = cv2.merge((B_channel, G_channel, R_channel))
    return RGB_mask

from PIL import Image


torch.no_grad()
def predict_segment(image, model, num_classes, device):
    PIL_image = Image.fromarray(image)
    tensor_image = transformer(PIL_image)
    tensor_image = tensor_image.to(device)

    pred_mask = model(torch.unsqueeze(tensor_image, dim=0))
    pred_mask = torch.argmax(pred_mask.squeeze(0).cpu(), dim=0)
    pred_mask = pred_mask.numpy()
    pred_mask = morpholocal_process(pred_mask, num_classes)
    rgb_mask = decode_segmap(pred_mask, num_classes)
    return rgb_mask


video_path = ()
cnt = 0
cap = cv2.VideoCapture(video_path)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        rgb_mask = predict_segment(frame, model, NUM_CLASSES, DEVICE)
        rgb_mask = cv2.resize(rgb_mask, dsize=frame.shape[:2])

        alpha = 0.6
        blend = cv2.addWeighted(frame, alpha, rgb_mask, 1 - alpha, 0)
        cv2.imshow('output', blend)

        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == ord('s'):
            cv2.waitKey(0)
    else:
        break
cap.release()
cv2.destroyAllWindows()

# Transfer Learning
# 한 분야의 문제를 해결하기 위해서 얻은 지식과 정보를 다른 문제를 푸는데 사용하는 방식 - 위키 백과
# 딥러닝 분야에서는 "이미지 분류" 문제를 해결하는데 사용했던 네트워크(DNN: Deep Neural Network)를 다른 데이터셋 또는 다른 문제(task)에 적용시켜 푸는 것을 의미

data_dir = './Medical/'
data_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
data_df.head()

def extract_client_id(x):
    return x.split('_')[0]

data_df['id'] = data_df.ImageId.apply(lambda x:extract_client_id(x))
data_df

regions = ['background', 'trachea', 'heart', 'lung']
colors = ((0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255))

def get_client_data(data_df, index):
    client_ids = np.unique(data_df.id.values)
    client_id = client_ids[index]
    client_data = data_df[data_df.id == client_id]
    image_files = list(client_data['ImageId'])
    mask_files = list(client_data['MaskId'])
    return client_id, image_files, mask_files


index = 1
client_id, image_files, mask_files = get_client_data(data_df, index)

canvas = np.zeros(shape=(512, 2 * 512 + 50, 3), dtype=np.uint8)

for i in range(len(image_files)):
    image = cv2.imread(os.path.join(data_dir, 'images', image_files[i]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(os.path.join(data_dir, 'masks', mask_files[i]))
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    num = 240
    mask[mask < num] = 0
    mask[mask >= num] = 255
    grid_pad = 50

    canvas[:, :512, :] = image
    canvas[:, 512 + grid_pad:2 * 512 + grid_pad, :] = mask

    text_buff = 410
    for j in range(1, len(regions)):
        cv2.putText(canvas, f'{regions[j].upper()}',
                    (900, text_buff), cv2.FONT_HERSHEY_SIMPLEX, 1, colors[j], 2)
        text_buff += 40
    cv2.imshow('CT frames', canvas)
    key = cv2.waitKey(60)
    if key == 27:
        break

