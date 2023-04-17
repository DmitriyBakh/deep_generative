# import matplotlib
# matplotlib.use('TkAgg',force=True)

# import matplotlib
# gui_env = ['TKAgg','GTKAgg','Qt4Agg','WXAgg']
# for gui in gui_env:
#     try:
#         print("testing", gui)
#         matplotlib.use(gui,warn=False, force=True)
#         from matplotlib import pyplot as plt
#         break
#     except:
#         continue
# print("Using:",matplotlib.get_backend())

# import matplotlib.pyplot as plt
# plt.switch_backend('newbackend')

# import matplotlib
# matplotlib.use('WXAgg',warn=False, force=True)
# from matplotlib import pyplot as plt
# print("Switched to:",matplotlib.get_backend())

# import matplotlib
# gui_env = [i for i in matplotlib.rcsetup.interactive_bk]
# non_gui_backends = matplotlib.rcsetup.non_interactive_bk
# print ("Non Gui backends are:", non_gui_backends)
# print ("Gui backends I will test for", gui_env)
# for gui in gui_env:
#     print ("testing", gui)
#     try:
#         matplotlib.use(gui,warn=False, force=True)
#         from matplotlib import pyplot as plt
#         print ("    ",gui, "Is Available")
#         plt.plot([1.5,2.0,2.5])
#         fig = plt.gcf()
#         fig.suptitle(gui)
#         plt.show()
#         print ("Using ..... ",matplotlib.get_backend())
#     except:
#         print ("    ",gui, "Not found")


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3

from scipy.stats import entropy
import scipy.misc
from scipy import linalg
import numpy as np
from tqdm import tqdm
from glob import glob
import pathlib
import os
import sys
import random

import torchvision

# from tqdm.notebook import tqdm
# !pip install git+https://github.com/S-aiueo32/lpips-pytorch.git

from lpips_pytorch import LPIPS, lpips
import matplotlib.pyplot as plt

dataset = torchvision.datasets.FashionMNIST('../data/fashion_mnist', download=True)

original_img_idx = 0
original_img, original_label = dataset[original_img_idx]
plt.imshow(original_img, cmap='gray')


# define as a criterion module (recommended)
criterion = LPIPS(
    net_type='alex',  # choose a network type from ['alex', 'squeeze', 'vgg']
    version='0.1'  # Currently, v0.1 is supported
)

def img2tensor(img):
    return torch.from_numpy(np.array(img.resize((512,512))))

np.random.seed(10)
img_indices = np.random.choice(np.arange(len(dataset)), 1000) # random images to be compared
img_indices = [idx for idx in img_indices if idx != original_img_idx]

distances, labels = [], []

# calculate LPIPS distances
for idx in tqdm(img_indices):
    # distance = # YOUR CODE HERE
    distance = criterion(img2tensor(original_img), img2tensor(dataset[idx][0]))
    distances.append(distance)
    labels.append(dataset[idx][1])


plt.figure(figsize=(17,10))
n_classes = 10
for label in range(n_classes):
    plt.subplot(4,3,label+1)
    plt.title(f'LPIPS. %s label (%d)' % ({True: 'Same', False: 'Another'}[label==original_label], label))
    plt.xlim((0.,0.7))
    plt.hist(np.array(distances)[np.array(labels)==label], bins=20, alpha=0.5);
plt.tight_layout();


plt.figure(figsize=(17,10))
closest_img_cnt = 9
closest_img_indices, closest_distances, closest_labels = [
    np.array(img_indices)[np.argsort(distances)[:closest_img_cnt]], 
    np.array(distances)[np.argsort(distances)[:closest_img_cnt]], 
    np.array(labels)[np.argsort(distances)[:closest_img_cnt]]]

for ax_idx, (img_idx, distance, label) in enumerate(zip(closest_img_indices, closest_distances, closest_labels)):
    img = np.array(dataset[img_idx][0])
    plt.subplot(3,3,ax_idx+1)
    plt.title(f'Label: %d Distance: %.3f'%(label, distance))
    plt.imshow(img, cmap='gray')
plt.tight_layout();
   

## GRADED PART, DO NOT CHANGE!
q3 = (np.array(closest_labels) == original_label).mean()
