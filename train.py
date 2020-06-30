import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from model.yoloLoss import YOLOLoss
from model.yolo import YoloBody
from utils.config import Config
from dataset.datasets import MaskData
from torch.utils.data import DataLoader
from dataset.datasets import collate_fn
from tqdm import tqdm


Cuda = True if torch.cuda.is_available() else False

def train(Epoch,Batch_Size, **kwargs):

    model = YoloBody(Config)
    train_data = MaskData("/Users/paulzyh/Desktop/yolo3/train.txt", train=True)
    # val_data = MaskData("val_file_path", train=False)
    train_dataloader = DataLoader(train_data, batch_size=Batch_Size, shuffle=True, collate_fn=collate_fn, num_workers=2)
    # val_dataloader = DataLoader(val_data, batch_size=Batch_Size, shuffle=True, num_workers=2)

    epoch_size = train_data.__len__() // Batch_Size
    # epoch_size_val = val_data.__len__() // Batch_Size

    learning_rate = 1e-4
    optimizer = optim.Adam(model.parameters(), learning_rate)

    #loss_function
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"],[-1,2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda))

    train_loss = 0
    # val_loss = 0

    loss_curve = []
    for epoch in range(Epoch):
        start_time = time.time()

        for ii, (images, targets) in enumerate(train_dataloader):

            if Cuda:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            else:
                images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]

            optimizer.zero_grad()
            outputs = model(images)

            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            train_loss += loss
            waste_time = time.time() - start_time
            print('\nEpoch:' + str(epoch + 1) + '/' + str(Epoch))
            print('iter:' + str(ii) + '/' + str(epoch_size) + ' || Total Loss: %.4f || %.4fs/step' % (train_loss / (ii + 1), waste_time))

            loss_curve.append(train_loss)
    torch.save(model.state_dict(), "yolo3_mask_detection.pth")

if __name__ == '__main__':
    train(1, 4)