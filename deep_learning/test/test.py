import os
import random
import torch
import torch.nn  as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchtext.data import TabularDataset
from torchtext import data
from torchtext.data import Iterator
from tqdm import tqdm
import shutil
import torch.nn.functional as F

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=3)
print(hypothesis)