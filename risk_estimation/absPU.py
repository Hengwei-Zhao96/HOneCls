import torch
import torch.nn as nn


def sigmoid_loss(x):
    return torch.sigmoid(-x)


class absNegative(nn.Module):
    def __init__(self, prior, loss='sigmoid', warm_up_epoch=-1):
        super(absNegative, self).__init__()
        self.prior = torch.tensor(prior)
        if torch.cuda.is_available():
            self.prior = self.prior.cuda()
        self.saved_loss = loss
        self.warm_up_epoch = warm_up_epoch
        self.loss = None

    def forward(self, pred, positive_mask, unlabeled_mask, epoch):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        if epoch < self.warm_up_epoch:
            self.loss = 'bce'
        else:
            self.loss = self.saved_loss

        if self.loss == 'sigmoid':
            positive_p_loss = sigmoid_loss(pred) * positive_mask
            unlabeled_loss = sigmoid_loss(-pred)
        else:
            NotImplemented

        positive_n_loss = unlabeled_loss * positive_mask
        unlabeled_n_loss = unlabeled_loss * unlabeled_mask

        estimated_p_loss = positive_p_loss.sum() / positive_mask.sum()

        estimated_u_n_loss = unlabeled_n_loss.sum() / unlabeled_mask.sum()
        estimated_p_n_loss = positive_n_loss.sum() / positive_mask.sum()

        estimated_n_loss = torch.abs((estimated_u_n_loss - self.prior * estimated_p_n_loss) / (1 - self.prior))

        loss = self.prior * estimated_p_loss + (1 - self.prior) * estimated_n_loss

        return loss, estimated_p_loss, estimated_n_loss, estimated_u_n_loss, estimated_p_n_loss
