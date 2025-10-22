import glob
import math
import os
import datetime
import argparse
import random

import pandas as pd
import torch
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
import torchvision.transforms as transforms
from networks import *
from ravdess import RAVDESSDataset
from mars_utils import train, validation, visulize_loss, visualize_accuracy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# define model input
def get_X(device, sample):
    images = sample["images"].to(device)
    images = images.permute(0, 2, 1, 3, 4)  # swap to be (N, C, D, H, W)
    mfcc = sample["mfcc"].to(device)
    n = images[0].size(0)
    return [images, mfcc], n

val_topk = (1, 2, 4,)
# Detect devices
use_cuda = torch.cuda.is_available()  # check if GPU exists
print(use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
loss_func = torch.nn.CrossEntropyLoss()

val_transform = {
        "image_transform": transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
        "audio_transform": None
    }

all_folder = sorted(list(glob.glob(os.path.join('RAVDESS/preprocessed', "Actor*"))))

# Data loading parameters
params = {'batch_size': 1, 'shuffle': True, 'num_workers': 0, 'pin_memory': True} \
    if use_cuda else {'batch_size': 1, 'shuffle': True, 'num_workers': 0}
val_fold = all_folder[4: 24]
val_set = RAVDESSDataset(val_fold, transform=val_transform)
val_loader = data.DataLoader(val_set, **params)
print("val fold: ")
print([os.path.basename(act) for act in val_fold])


video_model = resnet50(
            num_classes=8,
            shortcut_type='B',
            cardinality=32,
            sample_size=224,
            sample_duration=30
        )
audio_model = MFCCNet()
model_param = {
            "video": {
                "model": video_model,
                "id": 0
            },
            "audio": {
                "model": audio_model,
                "id": 1
            }
        }
multimodal_model = Mars_saveloss_Net(model_param)
multimodal_model.to(device)

model_path = os.path.join('result/marsloss_search_layer_no1/l=15/', 'fold_{}_search_best.pth'.format(1))
# model_path = 'result/fold_1_save_m3erloss_best.pth'
checkpoint = torch.load(model_path) if use_cuda else torch.load(model_path,
                                                                map_location=torch.device('cpu'))
multimodal_model.load_state_dict(checkpoint)
# epoch_test_loss, epoch_test_score,_,_ = validation(get_X, multimodal_model, device, loss_func, val_loader,
#                                                val_topk,11)
#
# print("Scores for each fold: ")
# print(epoch_test_score[0])
fea_v = []
fea_a = []
label = []
multimodal_model.eval()
with torch.no_grad():
    for sample in val_loader:
        # distribute data to device
        X, _ = get_X(device, sample)
        y = sample["emotion"].to(device).squeeze()
        label.append(y.detach().cpu().numpy())
        output_v, output_a, f_v, f_a = multimodal_model(X, y, 0, 0, 0)
        fea_v.append(f_v.detach().cpu().numpy())
        fea_a.append(output_a.detach().cpu().numpy())
fea_v = np.array(fea_v)
fea_v = np.squeeze(fea_v)
fea_a = np.array(fea_a)
fea_a = np.squeeze(fea_a)
label = np.array(label)
print(fea_v.shape,label.shape)


tsne = TSNE(n_components = 2)
digits_tsne = tsne.fit_transform(fea_a)
print(digits_tsne.shape)
di_all = np.concatenate((digits_tsne, label[:, np.newaxis]), axis=1)
df = pd.DataFrame(di_all)
print(di_all.shape)
df = pd.DataFrame(di_all)
# df.to_csv('plot/1/a_fea_m3.csv')

colors = ["#476A2A","#7851B8","#BD3430","#4A2D4E","#875525",
          "#A83683","#4E655E","#853541"]
font = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}
fig = plt.figure(figsize=(6,6))
ax = plt.subplot(111,facecolor='#FFFFFF')
ax.grid(False)
#设置坐标刻度值的大小以及刻度值的字体
plt.tick_params(labelsize=10)
labels = ax.get_xticklabels() + ax.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
# plt.figure(figsize=(8,8))
plt.xlim(digits_tsne[:,0].min(),digits_tsne[:,0].max()+1)
plt.ylim(digits_tsne[:,1].min(),digits_tsne[:,1].max()+1)
for i in range(len(label)):
    plt.text(digits_tsne[i,0],digits_tsne[i,1],str(label[i]),
             color = colors[label[i]],
             fontdict={'weight':'bold','size':12,'family':'Times New Roman'})
plt.xlabel('first',font)
plt.ylabel('second',font)
plt.show()
