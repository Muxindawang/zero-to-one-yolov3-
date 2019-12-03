from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2


def unique(tensor):
    """对tensor进行排序去重"""
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True):
    """
    计算prediction xywh....
    :param prediction:输出
    :param inp_dim: 输入图像尺寸
    :param anchors:
    :param num_classes:
    :param CUDA:
    :return:
    """
    batch_size = prediction.size(0)
    stride = inp_dim // prediction.size(2)   # 应该是缩小的倍数 416 // 13
    grid_size = inp_dim // stride     # 13
    bbox_attrs = 5 + num_classes     # 85
    num_anchors = len(anchors)      # 3

    prediction = prediction.view(batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    # prediction = prediction.view(batch_size, grid_size*grid_size*num_anchors, bbox_attrs)
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attrs)
    # 输入图像大, 检测图像小 相差stride倍 anchor放到特征图中应该缩小
    anchors = [(a[0] / stride, a[1] / stride) for a in anchors]   # anchors的高宽
    # 坐标处理
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # 网格偏移加到坐标中心
    grid = np.arange(grid_size)
    a, b = np.meshgrid(grid, grid)   # 生成grid*grid网格 ab就是网格的横纵坐标

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        prediction = prediction.cuda()
    # 预测的中心坐标相对于特征图左上角的网格偏移数
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    prediction[:, :, :2] += x_y_offset

    # 将锚框用于边界框的尺寸
    anchors = torch.FloatTensor(anchors)

    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size*grid_size, 1).unsqueeze(0)    # anchors的高宽
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4])*anchors  # bw bh

    prediction[:, :, 5:5+num_classes] = torch.sigmoid(prediction[:, :, 5:5+num_classes])  # 类别
    prediction[:, :, :4] *= stride   # 变到原图片尺寸大小的坐标

    return prediction


def write_results(prediction, confidence, num_classes, nms_conf=0.4):
    conf_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * conf_mask

    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = (prediction[:, :, 0] - prediction[:, :, 2] / 2)
    box_corner[:, :, 1] = (prediction[:, :, 1] - prediction[:, :, 3] / 2)
    box_corner[:, :, 2] = (prediction[:, :, 0] + prediction[:, :, 2] / 2)
    box_corner[:, :, 3] = (prediction[:, :, 1] + prediction[:, :, 3] / 2)
    prediction[:, :, :4] = box_corner[:, :, :4]

    batch_size = prediction.size(0)

    write = False

    for ind in range(batch_size):
        # ind批次索引
        image_pred = prediction[ind]
        # image_pred.shape    =    tensor([169*3,85]
        # confidence thresholding
        # nms
        max_conf, max_conf_score = torch.max(image_pred[:, 5:5 + num_classes], 1)
        # 找到类别分数最大的分数和index
        # max_conf 表示conf最高的分数. max_conf_score 表示最高分数的坐标索引   二者的shape都是 [169*3]
        max_conf = max_conf.float().unsqueeze(1)
        # shape tensor([169*3, 1])
        max_conf_score = max_conf_score.float().unsqueeze(1)
        # shape tensor([169*3, 1])
        seq = (image_pred[:, :5], max_conf, max_conf_score)
        # 0:4 xyxy 4:projectness 5:conf 6:类别分数最大的下标index
        image_pred = torch.cat(seq, 1)
        # shape tensor([169*3, 7])
        #               5               1          1
        #   169*3           +  169*3     + 169*3
        #

        non_zero_ind = (torch.nonzero(image_pred[:, 4]))
        # 找到非零索引
        # non_zero_ind . shape = tensor([169*3,1])    返回的是坐标索引
        # 如果有0  那么shape=  tensor([169*3-num(0),1])
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(), :].view(-1, 7)
            # 清除掉那些之前将其一行内容归零的行
        except:
            continue

        if image_pred_.shape[0] == 0:
            continue

            # 得到图片中检测到的类  未检测到的类别 [:,-1]=0 找到80个类别中检测到的类别 并排序
        img_classes = unique(image_pred_[:, -1])  # -1 index holds the class index
        # 对图片按照监测到的类别依次处理
        for cls in img_classes:
            # perform NMS
            # 获得一个特定类别的检测结果  cls_mask  该类别的mask
            cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
            # cls_mask.shape  =  tensor([169*3-num(0),1])
            # cls_mask 表示是cls这一类的检测框的信息 不是这一类的变为0
            class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()
            # 上一步只是对其归零,这一步删除整行
            # cls_mask_ind.shape  =  tensor([169*3-num(0)-num(cls!=cls)])   如:  [0,1,5,,10,65]表示代表该类别的是第0,1...行
            image_pred_class = image_pred_[class_mask_ind].view(-1, 7)
            # 取出 class_mask_inde 指示的行   shape = tensor([169*3-num(0)-num(cls!=cls), 7])

            # 对检测进行排序以便带有最大类的条目   根据分数排序
            conf_sort_index = torch.sort(image_pred_class[:, 4], descending=True)[1]
            # torch.sort返回的是一个元组(value,index)
            image_pred_class = image_pred_class[conf_sort_index]
            # 根据conf_sort_index序列(conf从大到小)返回指定行
            idx = image_pred_class.size(0)  # Number of detections

            """
                       以上代码 对预测的结果 先进行 类别阈值筛选(找到小于阈值的conf(第四个) 并归零 删除整行 返回image_pred_)
                       其次确定num(类别)  针对每一个类别 筛选该类别的边界框(到小于阈值的conf(第四个) 并归零 删除整行 返回image_pred_class)
                       最后根据conf进行排序
            """
            for i in range(idx):
                # Get the IOUs of all boxes that come after the one we are looking at
                # in the loop
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i + 1:])
                except ValueError:
                    break

                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i + 1:] *= iou_mask
                # 与conf最大的边界框的iou超过nm-thres就会被归零
                # remove the non_zero entries
                non_zero_ind = torch.nonzero(image_pred_class[:, 4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1, 7)

            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(
                ind)  # Repeat the batch_id for as many detections of the class cls in the image
            # 创建一列数据  image_pred_class.size(0)这么多行 一列 数据填充 ind batch指标
            seq = batch_ind, image_pred_class

            if not write:
                output = torch.cat(seq, 1)
                # 把 ind 和 后7列 cat
                write = True
            else:
                out = torch.cat(seq, 1)
                # 把前一类别的out和这一类别的out合并 按行叠加
                output = torch.cat((output, out))

    try:
        return output
    except:
        return 0


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes


    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def letterbox_image(img, inp_dim):
    """resize image with unchanged ascept ratio using padding"""
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w / img_w, h / img_h))
    new_h = int(img_h * min(w / img_w, h / img_h))
    resized_image = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w, :] = resized_image

    return canvas


def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network.

    Returns a Variable
    """
    img = (letterbox_image(img, (inp_dim, inp_dim)))
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)
    return img


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names



if __name__ == '__main__':
    pass

    # a = torch.tensor([[1,2,3],[4,5,6]])
    # print(a.size())
    # a = a.unsqueeze(0)
    # print(a.size())
    # b = a.squeeze()
    # print(b.size())
    # c = torch.cat((a, b),1)
    # print(c.size())
    # image_pred = torch.rand(2, 2)
    # print(image_pred[:, 5: 85])
    # max_conf, max_conf_score = torch.max(image_pred[:, 5:85], 1)
    # print(max_conf, max_conf_score)
    # cls = 5
    # image_pred_ = torch.tensor([[1,2,2],
    #                               [2,5,8]])
    # print(image_pred_[:,-2][1])
    # cls_mask = image_pred_ * (image_pred_[:, -1] == cls).float().unsqueeze(1)
    # print(cls_mask)
    # class_mask_ind = torch.nonzero(cls_mask[:, -2]).squeeze()  # 提取出该类别的index
    # image_pred_class = image_pred_[class_mask_ind].view(-1, 7)
