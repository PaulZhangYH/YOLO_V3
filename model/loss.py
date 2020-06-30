import numpy as np
import torch
import torch.nn as nn
from utils.bbox_iou import bbox_iou


def MSELoss(pred, target):
    return (pred - target) ** 2

def BCELoss(pred, target):
    epsilon = 1e-7
    output = -target * torch.log(pred) - (1-target) * torch.log(1.0 - pred)
    return output

class YOLOLoss(nn.Module):
    def __init__(self, anchors, num_classes):
        super(YOLOLoss, self).__init__()
        self.image_size = 416
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.bbox_attr = 5 + num_classes
        self.feature_length = [416//32, 416//16, 416//8]

        self.ignore_threshold = 0.5
        self.lambda_xy = 1.0
        self.lambda_wh = 1.0
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0


    def forward(self, pred, target):
        """

        :param pred: [batch_size,3*5+num_classes, 13, 13]
        :param target:[
        :return:
        """
        bs = pred.shape[0]

        # 计算每一个特征点对应原始图像上多少个像素
        in_h = pred.shpae[2]
        in_w = pred.shape[3]
        stride_h = self.image_size / in_h
        stride_w = self.image_size / in_w
        # 把先验框尺寸调整成对应特征图大小的形似
        scaled_anchors = [(anchor_w / stride_w, anchor_h / stride_h) for anchor_w, anchor_h in self.anchors]

        # pred:[batch_size, 3*(5+num_cls), 13, 13] -> [batch_size, 3, 13, 13, 5+num_cls]
        prediction = pred.view(bs, 3, self.bbox_attrs, in_h, in_w).permute(0,1,3,4,2).contiguous()

        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        w = prediction[..., 2]
        h = prediction[..., 3]
        conf = torch.sigmoid(prediction[..., 4])
        pred_cls = torch.sigmoid(prediction[..., 5:])

        # 找到哪些先验框内部包含物体
        mask, noobj_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y = self.get_target(target, scaled_anchors, in_w, in_h)

        noobj_mask = self.get_ignore(target, self.anchors, in_w, in_h)

        if torch.cuda.is_available():
            box_loss_scale_x = box_loss_scale_x.cuda()
            box_loss_scale_y = box_loss_scale_y.cuda()
            mask, noobj_mask = mask.cuda(), noobj_mask.cuda()
            tx, ty, tw, th = tx.cuda(), ty.cuda(), tw.cuda(), th.cuda()
            tconf, tcls = tconf.cuda(), tcls.cuda()

        box_loss_scale = 2 - box_loss_scale_y * box_loss_scale_x
        # loss
        loss_x = torch.sum(BCELoss(x, tx) / bs * box_loss_scale * mask)
        loss_y = torch.sum(BCELoss(y, ty) / bs * box_loss_scale * mask)
        loss_w = torch.sum(MSELoss(w, tw) / bs * box_loss_scale * mask)
        loss_h = torch.sum(MSELoss(h, th) / bs * box_loss_scale * mask)
        loss_conf = torch.sum(BCELoss(conf, mask) * mask / bs) + \
                    torch.sum(BCELoss(conf, mask) * noobj_mask / bs)

        loss_cls = torch.sum(BCELoss(pred_cls[mask == 1], tcls[mask == 1]) / bs)

        loss = loss_x * self.lambda_xy + loss_y * self.lambda_xy + \
               loss_w * self.lambda_wh + loss_h * self.lambda_wh + \
               loss_conf * self.lambda_conf + loss_cls * self.lambda_cls
        return loss

    def get_target(self, target, anchors, in_w, in_h):

        bs = len(target)

        anchor_index = [[0,1,2], [3,4,5], [6,7,8]][self.feature_length.index(in_w)]
        subtract_index = [0,3,6][self.feature_length.index(in_w)]

        # 创建全0和全1的矩阵，用来标记后续的有无物体
        mask = torch.zeros(bs, 3, in_h, in_w, requires_grad=False)
        no_mask = torch.ones(bs, 3, in_h, in_w, requires_grad=False)

        tx = torch.zeros(bs, 3, in_h, in_w, requires_grad=False)
        ty = torch.zeros(bs, 3, in_h, in_w, requires_grad=False)
        tw = torch.zeros(bs, 3, in_h, in_w, requires_grad=False)
        th = torch.zeros(bs, 3, in_h, in_w, requires_grad=False)
        tconf = torch.zeros(bs, 3, in_h, in_w, requires_grad=False)
        tcls = torch.zeros(bs, 3, in_h, in_w, requires_grad=False)

        box_loss_scale_x = torch.zeros(bs, 3, in_h, in_w, requires_grad=False)
        box_loss_scale_y = torch.zeros(bs, 3, in_h, in_w, requires_grad=False)

        for batch in range(bs):
            for t in range(target[batch].shape[0]):

                gx = target[batch][t, 0] * in_w
                gy = target[batch][t, 1] * in_h
                gw = target[batch][t, 2] * in_w
                gh = target[batch][t, 3] * in_h
                # 计算出属于哪个网格
                gi = int(gx)
                gj = int(gy)

                # 计算出真实框的位置
                gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
                # 计算出所有先验框的位置
                anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros(self.num_anchors, 2),
                                                                  np.array(anchors), 1)))

                # 计算重合程度: 将一个真实框与9个先验框分别计算iou
                anchor_iou = bbox_iou(gt_box, anchor_shapes)
                best_iou = np.argmax(anchor_iou)

                if best_iou not in anchor_index:
                    continue

                if (gj < in_h) and (gi < in_w):
                    best_iou = best_iou - subtract_index
                    # 判定哪些先验框存在物体
                    no_mask[batch, best_iou, gj, gi] = 0
                    mask[batch, best_iou, gj, gi] = 1

                    # 计算先验框中心调整参数,x,y,w,h的偏置
                    tx[batch, best_iou, gj, gi] = gx - gi
                    ty[batch, best_iou, gj, gi] = gy - gj
                    tw[batch, best_iou, gj, gi] = torch.log(gw / anchors[best_iou + subtract_index][0])
                    th[batch, best_iou, gj, gi] = torch.log(gh / anchors[best_iou + subtract_index][1])

                    # 用于获得 xywh 的比例
                    box_loss_scale_x[batch, best_iou, gj, gi] = target[batch][t, 2]
                    box_loss_scale_y[batch, best_iou, gj, gi] = target[batch][t, 3]
                    # 物体的置信度
                    tconf[batch, best_iou, gj, gi] = 1
                    # 种类
                    tcls[batch, best_iou, gj, gi] = 1

                else:
                    print('Step {0} out of bound'.format(batch))
                    print('gj: {0}, height: {1} | gi: {2}, width: {3}'.format(gj, in_h, gi, in_w))
                    continue
        return mask, no_mask, tx, ty, tw, th, tconf, tcls, box_loss_scale_x, box_loss_scale_y

    def get_ignore(self, prediction, target, scaled_anchors, in_w, in_h, noobj_mask):
        bs = len(target)
        anchor_index = [[0, 1, 2], [3, 4, 5], [6, 7, 8]][self.feature_length.index(in_w)]
        scaled_anchors = np.array(scaled_anchors)[anchor_index]
        # print(scaled_anchors)
        # 先验框的中心位置的调整参数
        x = torch.sigmoid(prediction[..., 0])
        y = torch.sigmoid(prediction[..., 1])
        # 先验框的宽高调整参数
        w = prediction[..., 2]  # Width
        h = prediction[..., 3]  # Height

        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor

        # 生成网格，先验框中心，网格左上角
        grid_x = torch.linspace(0, in_w - 1, in_w).repeat(in_w, 1).repeat(
            int(bs * self.num_anchors / 3), 1, 1).view(x.shape).type(FloatTensor)
        grid_y = torch.linspace(0, in_h - 1, in_h).repeat(in_h, 1).t().repeat(
            int(bs * self.num_anchors / 3), 1, 1).view(y.shape).type(FloatTensor)

        # 生成先验框的宽高
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))

        anchor_w = anchor_w.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(w.shape)
        anchor_h = anchor_h.repeat(bs, 1).repeat(1, 1, in_h * in_w).view(h.shape)

        # 计算调整后的先验框中心与宽高
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        for i in range(bs):
            pred_boxes_for_ignore = pred_boxes[i]
            pred_boxes_for_ignore = pred_boxes_for_ignore.view(-1, 4)

            for t in range(target[i].shape[0]):
                gx = target[i][t, 0] * in_w
                gy = target[i][t, 1] * in_h
                gw = target[i][t, 2] * in_w
                gh = target[i][t, 3] * in_h
                gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0).type(FloatTensor)

                anch_ious = bbox_iou(gt_box, pred_boxes_for_ignore, x1y1x2y2=False)
                anch_ious = anch_ious.view(pred_boxes[i].size()[:3])
                noobj_mask[i][anch_ious > self.ignore_threshold] = 0
                # print(torch.max(anch_ious))
        return noobj_mask






if __name__ == '__main__':
    data1 = [[1, 2], [3, 4], [5, 6]]
    data2 = [[1, 2], [3, 4], [5, 6]]

    t1 = torch.FloatTensor(data1)
    t2 = torch.FloatTensor(data2)

    print(t1 * (t2))









