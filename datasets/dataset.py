import os
from torch.utils.data import Dataset
from PIL import Image
from utils.letter_image import refine_image
from utils.refine_box import refine_box
import numpy as np

"""修改collate_fn，以适应边界框数量不同的targets"""
def collate_fn(batch):
    img, targets = zip(*batch) # transposed
    maxLen = 0
    for i in range(len(targets)):
        maxLen = max(maxLen, len(targets[i]))
    padding_element = np.array([0., 0., 0., 0., 0.])
    for target in targets:
        num_padding = maxLen - len(target)
        for i in range(num_padding):
            target = np.append(target, padding_element)
    return np.array(img), np.array(targets)

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



    def __getitem__(self, index):
        image_path = self.image[index]
        targets = self.target[index]

        boxes = []
        labels = []

        for target in targets:
            splited = target.split(',')
            box = splited[0:4]
            cls = splited[4:]
            boxes.append(box)
            labels.append(cls)

        image = Image.open(image_path)
        h, w, _ = image.shape
        image = refine_image(image)
        image = np.transpose(np.array(image) / 255., (2, 0, 1))
        boxes = refine_box(boxes)

        boxes[..., 0] = boxes[..., 0] / w
        boxes[..., 1] = boxes[..., 1] / h
        boxes[..., 2] = boxes[..., 2] / w
        boxes[..., 3] = boxes[..., 3] / h
        label = np.array(labels, dtype=int)
        targets = np.concatenate([boxes, label], 1)
        return image, targets


    def __len__(self):
        return self.sample


