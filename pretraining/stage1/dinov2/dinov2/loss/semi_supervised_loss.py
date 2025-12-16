# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F  
logger = logging.getLogger("dinov2")   

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, labels):
        # 计算 Softmax 概率
        probs = F.softmax(outputs, dim=1)
        # 计算 Focal Loss
        focal_loss = -self.alpha * (1 - probs) ** self.gamma * F.log_softmax(outputs, dim=1)
        # 取出对应标签的损失
        loss = focal_loss[range(len(labels)), labels]
        return loss

class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, gamma=2, alpha=0.25):
        super(CombinedLoss, self).__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.alpha = alpha
        self.focal_loss = FocalLoss(gamma, alpha)

    def forward(self, outputs, labels):
        # 仅保留有效标签
        masked_labels = labels#[mask]
        masked_outputs = outputs#[mask]
        if masked_labels.dtype != torch.long:
            masked_labels = masked_labels.long()
        # 计算交叉熵损失
        bce_loss = nn.CrossEntropyLoss(weight=self.class_weights, reduction='none')(masked_outputs, masked_labels)

        # 计算 Focal Loss
        #focal_loss = self.focal_loss(masked_outputs, masked_labels)

        # 计算有效样本的平均损失
        return bce_loss#.mean() + focal_loss.mean()