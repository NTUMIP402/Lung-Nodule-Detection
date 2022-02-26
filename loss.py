#!/usr/bin/python3
#coding=utf-8

import torch
from torch import nn
from torch.nn import functional as F



class Loss(nn.Module):
    def __init__(self, num_hard=0, class_loss='BCELoss', average=True):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard

        loss_dict = nn.ModuleDict(
            {'BCELoss': nn.BCELoss(),
             'MarginLoss': MarginLoss(size_average=average),
             'MSELoss': nn.MSELoss(reduction='mean' if average else 'sum'),
             'FocalMarginLoss': FocalMarginLoss(size_average=average)
             }
        )
        self.classify_loss = loss_dict[class_loss]

    @staticmethod
    def hard_mining(neg_output, neg_labels, num_hard):
        _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
        neg_output = torch.index_select(neg_output, 0, idcs)
        neg_labels = torch.index_select(neg_labels, 0, idcs)
        return neg_output, neg_labels

    def forward(self, output, labels, train=True):
        batch_size = labels.size(0)
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)

        # positive grids are labeled as 1
        pos_idcs = labels[:, 0] > 0.5
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        # negative grids are labeled as -1
        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]

        if self.num_hard > 0 and train:
            # Pick up the grid with the most wrong output (ie. highest output >> -1)
            neg_output, neg_labels = self.hard_mining(neg_output, neg_labels, self.num_hard * batch_size)

        neg_prob = neg_output

        if len(pos_output) > 0:
            pos_prob = pos_output[:, 0]

            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.item() for l in regress_losses]
            classify_loss = 0.5 * self.classify_loss(pos_prob, pos_labels[:, 0]) + \
                            0.5 * self.classify_loss(neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.9).sum()
            pos_total = len(pos_prob)

        else:
            regress_losses = [0, 0, 0, 0]
            classify_loss = 0.5 * self.classify_loss(neg_prob, neg_labels + 1)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0, 0, 0, 0]
        classify_loss_data = classify_loss.item()

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.9).sum()
        neg_total = len(neg_prob)
        
        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]


class Loss_recon(nn.Module):
    def __init__(self, num_hard=0, class_loss='MarginLoss', recon_loss_scale=1e-6, average=True):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()
        self.num_hard = num_hard
        self.recon_loss_scale = recon_loss_scale
        self.reconstruction_loss = nn.MSELoss(reduction='mean' if average else 'sum')

        loss_dict = nn.ModuleDict(
            {'BCELoss': nn.BCELoss(),
             'MarginLoss': MarginLoss(size_average=average),
             'MSELoss': nn.MSELoss(reduction='mean' if average else 'sum'),
             'FocalMarginLoss': FocalMarginLoss(size_average=average)
             }
        )
        self.classify_loss = loss_dict[class_loss]

    @staticmethod
    def hard_mining(neg_output, neg_labels, num_hard):
        _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)))
        neg_output = torch.index_select(neg_output, 0, idcs)
        neg_labels = torch.index_select(neg_labels, 0, idcs)
        return neg_output, neg_labels

    def forward(self, output, labels, images, reconstructions, train=True):
        batch_size = labels.size(0)
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)

        # positive grids are labeled as 1
        pos_idcs = labels[:, 0] > 0.5
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        # negative grids are labeled as -1
        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]

        if self.num_hard > 0 and train:
            # Pick up the grid with the most wrong output (ie. highest output >> -1)
            neg_output, neg_labels = self.hard_mining(neg_output, neg_labels, self.num_hard * batch_size)

        neg_prob = neg_output
        
        if len(pos_output) > 0:  # there are positive anchors in this crop
            pos_prob = pos_output[:, 0]
           
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.item() for l in regress_losses]
            classify_loss = .5 * self.classify_loss(pos_prob, pos_labels[:, 0]) + \
                            .5 * self.classify_loss(neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)

        else:
            regress_losses = [0, 0, 0, 0]
            classify_loss = 0.5 * self.classify_loss(neg_prob, neg_labels + 1)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0, 0, 0, 0]
        classify_loss_data = classify_loss.item()

        # Total loss = classify loss + regress_loss of z, h, w, d + recon_loss * recon_loss_scale
        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        reconstruction_loss = self.recon_loss_scale * self.reconstruction_loss(reconstructions, images)
        reconstruction_loss_data = reconstruction_loss.item()
        loss += reconstruction_loss

        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)
        
        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total] + [reconstruction_loss_data]
       
class FocalLoss(nn.Module):
    """ Kaiming's Focal loss
    https://arxiv.org/pdf/1708.02002.pdf
    """
    def __init__(self, num_classes=1, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.num_classes = num_classes  # exclude the background
        self.alpha = alpha
        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def one_hot_embedding(self, labels, num_classes):
        '''Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N,#classes].
        '''
        y = torch.eye(num_classes)  # [D,D]
        return y[labels]  # [N,D]

    def focal_loss(self, x, y):
        '''Focal loss.

        Args:
          x: (tensor) sized [N] --> [N, 1].
          y: (tensor) sized [N].

        Return:
          (tensor) focal loss.
        '''
        n = x.size(0)
        x = x.view(n, -1)  # (N, 1)

        # Convert the label to one-hot encoded target
        t = self.one_hot_embedding(y.type(torch.long), 1 + self.num_classes)
        t = t[:, 1:]  # exclude background
        t = t.view(n, -1)
        t = t.to(self.device)  # [N, num_classes]

        # Calculate weight from target and prediction distribution
        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)         # pt = p if t = 1 else 1-p
        w = self.alpha * t + (1 - self.alpha) * (1 - t)  # w = alpha if t = 1 else 1-alpha
        w = w * (1 - pt).pow(self.gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def forward(self, output, labels):
        output = output.view(-1, 1)
        labels = labels.view(-1, 1)
        classify_loss = self.focal_loss(output, labels)
        return classify_loss


class MarginLoss(nn.Module):
    def __init__(self, num_classes=1, size_average=True, loss_lambda=0.5):
        '''
        Margin loss for digit existence
        Eq. (4): L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2

        Args:
            size_average: should the losses be averaged (True) or summed (False) over observations for each minibatch.
            loss_lambda: parameter for down-weighting the loss for missing digits
            num_classes: number of classes (exclude the background)
        '''
        super().__init__()
        self.size_average = size_average
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.loss_lambda = loss_lambda
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def one_hot_embedding(self, labels, num_classes):
        '''Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N, #classes].
        '''
        y = torch.eye(num_classes)  # [D,D]
        return y[labels]  # [N,D]

    def forward(self, inputs, labels):
        """
        :param inputs: [n, num_classes] with one-hot encoded
        :param labels: [n,]
        """
        n = inputs.size(0)
        inputs = inputs.view(n, -1)

        # Convert the label to one-hot encoded target
        labels = self.one_hot_embedding(labels.type(torch.long), 1 + self.num_classes)
        labels = labels[:, 1:]  # exclude background
        labels = labels.view(n, -1)
        labels = labels.to(self.device)  # (N, num_classes)

        left = labels * F.relu(self.m_plus - inputs)**2
        right = self.loss_lambda * (1 - labels) * F.relu(inputs - self.m_minus)**2
        L_k = left + right

        # Summation of all classes
        L_k = L_k.sum(dim=-1)

        if self.size_average:
            return L_k.mean()
        else:
            return L_k.sum()


class FocalMarginLoss(nn.Module):
    def __init__(self, num_classes=1, alpha=0.25, gamma=2, size_average=True):
        '''
        Focal loss binds with Margin loss for one-hot label
        Eq. (4): L_k = T_k * max(0, m+ - ||v_k||)^2 + lambda * (1 - T_k) * max(0, ||v_k|| - m-)^2

        Args:
            size_average: should the losses be averaged (True) or summed (False) over observations for each minibatch.
            loss_lambda: parameter for down-weighting the loss for missing digits
            num_classes: number of classes (exclude the background)
        '''
        super().__init__()
        self.size_average = size_average
        self.m_plus = 0.9
        self.m_minus = 0.1
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def one_hot_embedding(self, labels, num_classes):
        '''Embedding labels to one-hot form.

        Args:
          labels: (LongTensor) class labels, sized [N,].
          num_classes: (int) number of classes.

        Returns:
          (tensor) encoded labels, sized [N,#classes].
        '''
        y = torch.eye(num_classes)  # [D,D]
        return y[labels]  # [N,D]

    def forward(self, inputs, labels):
        """
        :param inputs: (n, num_classes) with one-hot encoded
        :param labels: (n,)
        :return: loss value
        """
        n = inputs.size(0)
        inputs = inputs.view(n, -1)

        # Convert the label to one-hot encoded target
        t = self.one_hot_embedding(labels.type(torch.long), 1 + self.num_classes)
        t = t[:, 1:]  # exclude background
        t = t.view(n, -1)
        t = t.to(self.device)  # (N, num_classes)

        # Calculate weight from target and prediction distribution
        p = inputs  # capsule's output has already squashed to 0-1, so sigmoid is not needed.
        pt = p * t + (1 - p) * (1 - t)                   # pt = p if t = 1 else 1-p
        w = self.alpha * t + (1 - self.alpha) * (1 - t)  # w = alpha if t = 1 else 1-alpha
        w = w * (1 - pt).pow(self.gamma)

        labels = labels.view(n, 1)
        left = labels * F.relu(self.m_plus - inputs)**2
        right = w * (1 - labels) * F.relu(inputs - self.m_minus)**2
        L_k = left + right

        # Summation of all classes
        L_k = L_k.sum(dim=-1)

        if self.size_average:
            return L_k.mean()
        else:
            return L_k.sum()



