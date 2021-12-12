#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：mask_rcnn_pytorch 
@File    ：train.py
@Author  ：zhuofalin
@Date    ：2021/11/24 21:19 
'''

import torch, os
from my_dataset import PenFudanDataset
from utils.models import get_model_instance_segmentation, get_transform
from utils.engine import train_one_epoch, evaluate
from utils import utils


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # define weight_save path
    weight_save_path = 'weights'
    weight_save_name = 'mask_R_CNN_model.pth'

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PenFudanDataset('E:\\PennFudanPed\\', get_transform(train=True))
    dataset_test = PenFudanDataset('E:\\PennFudanPed\\', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 60

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    # save weight
    # 这里有一处不足之处：当训练结束时就会保存权重，而在实际实验中可能会需要针对某些指标（精度 dice等）再考虑是否保存
    # 该类情况请自行调整
    torch.save(model.state_dict(), os.path.join(weight_save_path, weight_save_name))
    print("That's it!")

if __name__ == "__main__":
    main()