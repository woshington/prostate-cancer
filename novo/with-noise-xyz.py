#%%
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler
from warmup_scheduler import GradualWarmupScheduler
import albumentations
from work.utils.dataset import RGB2XYZTransform, PandasDataset
from work.utils.models import EfficientNet
from work.utils.train import train_model
from work.utils.metrics import model_checkpoint
import random
#%%
backbone_model = 'efficientnet-b0'
pretrained_model = {
    backbone_model: '../pre-trained-models/efficientnet-b0-08094119.pth'
}

data_dir = '../../dataset'
images_dir = os.path.join(data_dir, 'tiles')

df_train = pd.read_csv(f"../data/train_5fold.csv")
#%%
seed = 42
shuffle = True
batch_size = 2
num_workers = 4
output_classes = 5
init_lr = 3e-4
warmup_factor = 10
warmup_epochs = 1
n_epochs = 30
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
loss_function = nn.BCEWithLogitsLoss()

torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

#%%
transforms = albumentations.Compose([
    RGB2XYZTransform(),
    albumentations.Transpose(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.HorizontalFlip(p=0.5),
])

valid_transforms =albumentations.Compose([
    RGB2XYZTransform()
])
#%%
df_train.columns = df_train.columns.str.strip()

train_indexes = np.where((df_train['fold'] != 3))[0]
valid_indexes = np.where((df_train['fold'] == 3))[0]

train = df_train.loc[train_indexes]
valid = df_train.loc[valid_indexes]

train_dataset = PandasDataset(images_dir, train, transforms=transforms)
valid_dataset = PandasDataset(images_dir, valid, transforms=valid_transforms)
#%%
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=2, num_workers=num_workers, sampler = RandomSampler(train_dataset)
)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=2, num_workers=num_workers, sampler = RandomSampler(valid_dataset)
)
#%%
import matplotlib.pyplot as plt

data_iter = iter(train_loader)
images, labels, ids = next(data_iter)

def denormalize(image):
    image = image.numpy().transpose((1, 2, 0))
    image = (image - image.min()) / (image.max() - image.min())  # Normalizar para [0, 1]
    return image

fig, axes = plt.subplots(1, batch_size, figsize=(15, 5))
for i in range(batch_size):
    img = denormalize(images[i])
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Label: {ids[i]}")

plt.show()

#%%
model = EfficientNet(
    backbone=backbone_model,
    output_dimensions=output_classes,
    pre_trained_model=pretrained_model
)
model = model.to(device)
#%%
optimizer = optim.Adam(model.parameters(), lr = init_lr / warmup_factor)
scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs - warmup_epochs)
scheduler = GradualWarmupScheduler(optimizer, multiplier = warmup_factor, total_epoch = warmup_epochs, after_scheduler=scheduler_cosine)
save_path = f'models/with-noise-xyz.pth'
#%%
train_model(
    model=model,
    epochs=n_epochs,
    optimizer=optimizer,
    scheduler=scheduler,
    train_dataloader=train_loader,
    valid_dataloader=valid_loader,
    checkpoint=model_checkpoint,
    device=device,
    loss_function=loss_function,
    path_to_save_metrics="logs/with-noise-xyz.txt",
    path_to_save_model=save_path
)