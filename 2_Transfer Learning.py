# Transfer Learning
# 한 분야의 문제를 해결하기 위해서 얻은 지식과 정보를 다른 문제를 푸는데 사용하는 방식 - 위키 백과
# 딥러닝 분야에서는 "이미지 분류" 문제를 해결하는데 사용했던 네트워크(DNN: Deep Neural Network)를 다른 데이터셋 또는 다른 문제(task)에 적용시켜 푸는 것을 의미

import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

data_dir = './MEDICAL-DATASET/'
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

# 데이터셋 구축과 연산을 위한 텐서변환 모듈 작성하기
import torch
import torch.nn as nn

IMAGE_SIZE = 224


class CT_dataset():
    def __init__(self, data_dir, phase, transformer=None):
        self.phase = phase
        self.image_dir = os.path.join(data_dir, phase, 'images')
        self.mask_dir = os.path.join(data_dir, phase, 'masks')
        self.image_files = [filename for filename in os.listdir(self.image_dir)
                            if filename.endswith('jpg')]
        self.mask_files = [filename for filename in os.listdir(self.mask_dir)
                           if filename.endswith('jpg')]
        self.transformer = transformer

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.image_dir, self.image_files[index]))
        image = cv2.resize(image, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
        mask = cv2.imread(os.path.join(self.mask_dir, self.mask_files[index]))
        mask = cv2.resize(mask, dsize=(IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask[mask < 240] = 0
        mask[mask >= 240] = 255
        mask = mask / 255

        mask_H, mask_W, mask_C = mask.shape
        background = np.ones(shape=(mask_H, mask_W))
        background[mask[..., 0] != 0] = 0
        background[mask[..., 1] != 0] = 0
        background[mask[..., 2] != 0] = 0

        mask = np.concatenate([np.expand_dims(background, -1), mask], axis=-1)
        mask = np.argmax(mask, axis=-1)

        if self.transformer:
            image = self.transformer(image)

        target = torch.from_numpy(mask).long()
        return image, target

from torchvision import transforms

def build_transformer():
    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transformer


def collate_fn(batch):
    images = []
    targets = []
    for a, b in batch:
        images.append(a)
        targets.append(b)
    images = torch.stack(images, dim=0)
    targets = torch.stack(targets, dim=0)

    return images, targets

data_dir = './MEDICAL-DATASET/'
transformer = build_transformer()
dset = CT_dataset(data_dir=data_dir, phase='train', transformer=transformer)

image, target = dset[0]
print(f'image shape: {image.shape}')
print(f'masks shape: {target.shape}')

from torch.utils.data import DataLoader

dloader = DataLoader(dset, batch_size=4, shuffle=True, collate_fn=collate_fn)

for index, batch in enumerate(dloader):
    images = batch[0]
    targets = batch[1]
    print(f'images shape: {images.shape}')
    print(f'masks shape: {targets.shape}')
    if index == 0:
        break


def build_dataloader(data_dir, batch_size=4):
    transformer = build_transformer()
    dataloaders = {}
    train_dataset = CT_dataset(data_dir=data_dir, phase='train', transformer=transformer)
    dataloaders['train'] = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = CT_dataset(data_dir=data_dir, phase='val', transformer=transformer)
    dataloaders['val'] = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    return dataloaders

data_dir = './MEDICAL-DATASET/'
dataloaders = build_dataloader(data_dir=data_dir)

for phase in ['train', 'val']:
    for index, batch in enumerate(dataloaders[phase]):
        images = batch[0]
        tragets = batch[1]
        print(f'images shape: {images.shape}')
        print(f'targets shape: {targets.shape}')

        if index == 0:
            break

# VGG16 Backbone을 이용한 U-Net 아키텍처 구현해보기
# * Backbone: 입력 이미지를 feature map으로 변형시켜주는 부분. ImageNet 데이터셋으로 pre-trained 시킨 VGG16, ResNet-50등이 대표적

from torchvision import transforms, models

def ConvLayer(in_channels, out_channels, kernel_size=3, padding=1):
    layers = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return layers

def UpConvLayer(in_channels, out_channels):
    layers = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return layers

from torchvision import models

vgg16 = models.vgg16_bn(pretrained=False)

vgg16


class Encoder(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        backbone = models.vgg16_bn(pretrained=pretrained).features
        self.conv_block1 = nn.Sequential(*backbone[:6])
        self.conv_block2 = nn.Sequential(*backbone[6:13])
        self.conv_block3 = nn.Sequential(*backbone[13:20])
        self.conv_block4 = nn.Sequential(*backbone[20:27])
        self.conv_block5 = nn.Sequential(*backbone[27:34], ConvLayer(512, 1024, kernel_size=1, padding=0))

    def forward(self, x):
        encode_features = []
        out = self.conv_block1(x)
        encode_features.append(out)

        out = self.conv_block2(out)
        encode_features.append(out)

        out = self.conv_block3(out)
        encode_features.append(out)

        out = self.conv_block4(out)
        encode_features.append(out)

        out = self.conv_block5(out)
        return out, encode_features

encoder = Encoder(pretrained=False)
x = torch.randn(1, 3, 224, 224)
out, ftrs = encoder(x)

for ftr in ftrs:
    print(ftr.shape)
print(out.shape)


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
        out = torch.cat([out, encoder_features[-1]], dim=1)
        out = self.conv_block1(out)

        out = self.upconv_layer2(out)
        out = torch.cat([out, encoder_features[-2]], dim=1)
        out = self.conv_block2(out)

        out = self.upconv_layer3(out)
        out = torch.cat([out, encoder_features[-3]], dim=1)
        out = self.conv_block3(out)

        out = self.upconv_layer4(out)
        out = torch.cat([out, encoder_features[-4]], dim=1)
        out = self.conv_block4(out)
        return out

encoder = Encoder(pretrained=False)
decoder = Decoder()
x = torch.randn(1, 3, 224, 224)
out, ftrs = encoder(x)
out = decoder(out, ftrs)

print(out.shape)


class UNet(nn.Module):
    def __init__(self, num_classes, pretrained):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.decoder = Decoder()
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        out, encode_features = self.encoder(x)
        out = self.decoder(out, encode_features)
        out = self.head(out)
        return out

model = UNet(num_classes=4, pretrained=False)
x = torch.randn(1, 3, 224, 224)
out = model(x)

print(out.shape)

# Semantic segmentation Loss와 학습코드 작성하기

import torch.nn.functional as F


class UNet_metric():
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.CE_loss = nn.CrossEntropyLoss(reduction='mean')

    def __call__(self, pred, target):
        loss1 = self.CE_loss(pred, target)
        onehot_pred = F.one_hot(torch.argmax(pred, dim=1), num_classes=self.num_classes).permute(0, 3, 1, 2)
        onehot_target = F.one_hot(target, num_classes=self.num_classes).permute(0, 3, 1, 2)
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


def train_one_epoch(dataloaders, model, criterion, optimizer, device):
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

            if index == 10:  # index * mini_batch 데이터 수 만큼 데이터를 한정
                break

        losses[phase] = running_loss / index
        dice_coefficients[phase] = running_dice_coeff / index

    return losses, dice_coefficients

# Weight Initialization 과 Transfer learning 모델 비교하기

### 가중치 초기화
# * 신경망 학습에서 가중치의 초깃값을 무엇으로 설정하느냐가 학습에 많은 영향을 줌
# * 뉴런의 가중치를 기반으로 error를 결정하기 때문

### He Initialization
# * ReLU를 위해 만들어진 초기화 방법
# * activation function으로 ReLU를 사용하면 He 초깃값을 쓰는것이 좋음
# * gradient의 vanishing, exploding 문제를 완화

# isinstance(): 해당 객체가 일치하는지 확인
# isinatance(1, int) -> True
def He_initialization(module):
    if isinstance(module, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(module.weight) # He Initialization
    elif isinstance(module, torch.nn.BatchNorm2d):
        module.weight.data.fill_(1.0)

data_dir = './MEDICAL-DATASET/'
NUM_CLASSES = 4
IMAGE_SIZE = 224
BATCH_SIZE = 12
# is_cuda = True
# DEVICE = torch.device('cuda' if torch.cuda.is_available() and is_cuda else 'cpu')
DEVICE = 'cpu'

dataloaders = build_dataloader(data_dir, batch_size=BATCH_SIZE)
model = UNet(num_classes=NUM_CLASSES, pretrained=False)
model.apply(He_initialization)
model = model.to(DEVICE)
criterion = UNet_metric(num_classes=NUM_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 30

train_loss_def, train_dice_coefficient_def = [], []
val_loss_def, val_dice_coefficient_def = [], []

for epoch in range(num_epochs):
    losses, dice_coefficients = train_one_epoch(dataloaders, model, criterion, optimizer, DEVICE)
    train_loss_def.append(losses['train'])
    val_loss_def.append(losses['val'])
    train_dice_coefficient_def.append(dice_coefficients['train'])
    val_dice_coefficient_def.append(dice_coefficients['val'])

    print(f"{epoch}/{num_epochs} - Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}, " + \
          f" Train dice: {dice_coefficients['train']:.4f}, Val dice: {dice_coefficients['val']:.4f}")

### Weight transfer pre-trained on ImageNet

data_dir = './MEDICAL-DATASET/'
NUM_CLASSES = 4
IMAGE_SIZE = 224
BATCH_SIZE = 12
# is_cuda = True
# DEVICE = torch.device('cuda' if torch.cuda.is_available() and is_cuda else 'cpu')
DEVICE = 'cpu'

dataloaders = build_dataloader(data_dir, batch_size=BATCH_SIZE)
model = UNet(num_classes=NUM_CLASSES, pretrained=True)
model = model.to(DEVICE)
criterion = UNet_metric(num_classes=NUM_CLASSES)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 30

train_loss_prt, train_dice_coefficient_prt = [], []
val_loss_prt, val_dice_coefficient_prt = [], []

for epoch in range(num_epochs):
    losses, dice_coefficients = train_one_epoch(dataloaders, model, criterion, optimizer, DEVICE)
    train_loss_prt.append(losses['train'])
    val_loss_prt.append(losses['val'])
    train_dice_coefficient_prt.append(dice_coefficients['train'])
    val_dice_coefficient_prt.append(dice_coefficients['val'])

    print(f"{epoch}/{num_epochs} - Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}, " + \
          f" Train dice: {dice_coefficients['train']:.4f}, Val dice: {dice_coefficients['val']:.4f}")

### Weight transfer with freezing encoder layer

data_dir = './MEDICAL-DATASET/'
NUM_CLASSES = 4
IMAGE_SIZE = 224
BATCH_SIZE = 12
# is_cuda = True
# DEVICE = torch.device('cuda' if torch.cuda.is_available() and is_cuda else 'cpu')
DEVICE = 'cpu'

dataloaders = build_dataloader(data_dir, batch_size=BATCH_SIZE)
model = UNet(num_classes=NUM_CLASSES, pretrained=True)
model = model.to(DEVICE)
model.encoder.requires_grad_ = False
criterion = UNet_metric(num_classes=NUM_CLASSES)
optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters(), lr=0.001, momentum=0.9)

num_epochs = 30

train_loss_frz, train_dice_coefficient_frz = [], []
val_loss_frz, val_dice_coefficient_frz = [], []

for epoch in range(num_epochs):
    losses, dice_coefficients = train_one_epoch(dataloaders, model, criterion, optimizer, DEVICE)
train_loss_frz.append(losses['train'])
val_loss_frz.append(losses['val'])
train_dice_coefficient_frz.append(dice_coefficients['train'])
val_dice_coefficient_frz.append(dice_coefficients['val'])

print(f"{epoch}/{num_epochs} - Train loss: {losses['train']:.4f}, Val loss: {losses['val']:.4f}, " + \
      f" Train dice: {dice_coefficients['train']:.4f}, Val dice: {dice_coefficients['val']:.4f}")