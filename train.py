from pickle import TRUE
from turtle import width
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_img,
# )

#HYPERPARAMETERS ETC
LEARNING_RATE = 1E-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2 
IMAGE_HEIGHT = 160 #1280 originally
IMAGE_WIDTH = 240 #1918 originally
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = 'data/train_images/'
TRAIN_MASK_DIR = 'data/train_masks/'
VAL_IMG_DIR = 'data/val_images/'
VAL_MASK_DIR = 'data/val_masks/'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data,targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)


        #forward
        with torch.cuda.amp.autocast():
            predictions= model(data)
            loss = loss_fn(predictions, targets)


        #backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.ster(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
    train_transform = A.compose([
        A.Resize(height=IMAGE_HEIGHT,width= IMAGE_WIDTH),
        A.Rotate(limit=35,p=1.0),
        A.Horizontalflip(p=0.5),
        A.Verticalflip(p=0.1),
        #ToTensor doesn't divide by 255 like Pytorch
        #its done inside normalize function
        A.Normalize(
            mean =[0.0,0.0,0.0],
            std = [1.0,1.0,1.0],
            max_pixel_value =255.0,

        ),
        ToTensorV2(),
    ])

if __name__ =='__main__':
    main()