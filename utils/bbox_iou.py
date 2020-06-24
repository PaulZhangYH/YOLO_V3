import torch

def bbox_iou(box1, box2):
    box1_x1, box2_x1 = box1[:, 0] - box1[:, 2] / 2, box2[:, 0] - box2[:, 2] / 2
    box1_y1, box2_y1 = box1[:, 1] - box1[:, 3] / 2, box2[:, 1] - box2[:, 3] / 2
    box1_x2, box2_x2 = box1[:, 0] + box1[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
    box1_y2, box2_y2 = box1[:, 1] + box1[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_x1, inter_y1 = torch.min(box1_x1, box2_x1), torch.min(box2_y1, box2_y1)
    inter_x2, inter_y2 = torch.min(box1_x2, box2_x2), torch.min(box1_y2, box2_y2)

    inter_area = torch.clamp((inter_x2 - inter_x1 + 1), min=0) * torch.clamp((inter_y2 - inter_y1 + 1), min=0)
    box1_area = (box1_x2 - box1_x1 + 1) * (box1_y2 - box1_y1 + 1)
    box2_area = (box2_x2 - box2_x1 + 1) * (box2_y2 - box2_y1 + 1)

    iou = inter_area / (box1_area + box2_area - inter_area + 1e-7)
    return iou



