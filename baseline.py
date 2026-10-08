import os
import time

import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import ConcatDataset
from sklearn.metrics import f1_score

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

input_size = 500
num_classes = 4

num_epochs = 1
BATCH_SIZE = 50
lr = 1e-3

img_happy_path = os.listdir("original_data/EMOIMG_기쁨_SAMPLE")
img_surprise_path = os.listdir("original_data/EMOIMG_당황_SAMPLE")
img_angry_path = os.listdir("original_data/EMOIMG_분노_SAMPLE")
img_sad_path = os.listdir("original_data/EMOIMG_상처_SAMPLE")
len_happy = len(img_happy_path)
len_surprise = len(img_surprise_path)
len_angry = len(img_angry_path)
len_sad = len(img_sad_path)

### Custom Dataset ###
class CustomDataset(Dataset):
    def __init__(self, transform, mode='train'):
        self.transform = transform
        self.mode = mode
        self.sentiments = ["기쁨", "당황", "분노", "상처"]
        self.sentiment_label = {"기쁨":0, "당황":1, "분노":2, "상처":3}
        
        if self.mode == 'train':
            self.image_folder = img_happy_path[:int(len_happy*0.8)]+img_surprise_path[:int(len_surprise*0.8)] \
                            +img_angry_path[:int(len_angry*0.8)]+img_sad_path[:int(len_sad*0.8)]
        elif self.mode == 'test':
            self.image_folder = img_happy_path[int(len_happy*0.8):]+img_surprise_path[int(len_surprise*0.8):] \
                            +img_angry_path[int(len_angry*0.8):]+img_sad_path[int(len_sad*0.8):]

    def __len__(self):
        return len(self.image_folder)
    
    def __getitem__(self, idx):
        img_path = self.image_folder[idx]
        for sentiment in self.sentiments:
            try:
                img = Image.open(f"original_data/EMOIMG_{sentiment}_SAMPLE/{img_path}").convert("RGB")
                img = self.transform(img)
            
                label = self.sentiment_label[sentiment]
            except:
                pass
        return (img, label)

transforms_train = transforms.Compose([transforms.Resize((input_size, input_size)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)), ])
transforms_test = transforms.Compose([transforms.Resize((input_size, input_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.1307, 0.1307, 0.1307), (0.3081, 0.3081, 0.3081)),])

train_set = CustomDataset(mode='train', transform=transforms_train)
test_set = CustomDataset(mode='test', transform=transforms_test)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

print('Loaded dataloader')

# ### Model / Optimizier ###
# model = models.resnet18(pretrained=True)
# ### Transfer Learning ###
# num_features = model.fc.in_features
# model.fc = nn.Linear(num_features, num_classes)

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
model = AlexNet()

model.to(device)

loss_function = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for epoch in range(num_epochs):
    correct = 0
    loss_sum = 0
    for i, (image, target) in enumerate(train_loader):
        image, target = image.to(device), target.to(device)
        output = model(image)
        
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
        
        optimizer.zero_grad()
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        
    print(f"{epoch} accuracy: {100*correct/len(train_loader.dataset)}\
         loss: {loss_sum/len(train_loader.dataset)}")

model.eval()
correct = 0
for i, (image, target) in enumerate(test_loader):
    image, target = image.to(device), target.to(device)
    output = model(image)
    
    pred = output.data.max(1)[1]
    correct += pred.eq(target.view_as(pred)).sum().item()
        
print(f"Test Accuracy: {100*correct/len(test_loader.dataset)}")