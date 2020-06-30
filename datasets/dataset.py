import os
from torch.utils.data import Dataset
from PIL import Image
from utils.letter_image import refine_image
from utils.refine_box import refine_box
import numpy as np


class Datasets(Dataset):
    def __init__(self, file_path):
        self.image = []
        self.target = []

        assert os.path.isfile(file_path), 'File not found %s.' %(file_path)

        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                splited = line.strip().split()
                if(len(splited[1:]) == 0):
                    continue
                self.image.append(splited[0])
                self.target.append(splited[1:])

        self.sample = len(self.image)



    def __getitem__(self, item):
        image_path = self.image[item]
        targets = self.target[item]
        boxes = []
        labels = []
        for target in targets:
            splited =  target.split(',')
            box = splited[0:4]
            label = splited[4:]
            boxes.append(box)
            labels.append(label)

        image = Image.open(image_path)
        w, h = image.size
        # if self.train:
        #     image, boxes = random_blur(image, boxes)

        image = refine_image(image, 416)
        image = np.transpose(np.array(image)/255., (2,0,1))

        # 边界框中心、归一化
        boxes = refine_box(boxes)
        boxes = np.array(boxes)
        boxes[..., 0] = boxes[..., 0] / h
        boxes[..., 1] = boxes[..., 1] / w
        boxes[..., 2] = boxes[..., 2] / w
        boxes[..., 3] = boxes[..., 3] / h
        labels = np.array(labels, dtype=int)
        targets = np.concatenate([boxes, labels], 1)
        return image, targets

    def __len__(self):
        return self.sample


