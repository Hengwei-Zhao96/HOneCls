import torch
import torch.nn as nn

from risk_estimation.loss_function import sigmoid_loss, bce_loss


class OneClassRiskEstimator(nn.Module):
    def __init__(self, prior, class_weight=0.5, focal_weight=2, loss='sigmoid', warm_up_epoch=50):
        super(OneClassRiskEstimator, self).__init__()
        self.prior = torch.tensor(prior)
        self.class_weight = torch.tensor(class_weight)
        if torch.cuda.is_available():
            self.prior = self.prior.cuda()
            self.class_weight = self.class_weight.cuda()
        self.focal_weight = focal_weight
        self.saved_loss = loss
        self.warm_up_epoch = warm_up_epoch
        self.loss = None

    def forward(self, pred, positive_mask, unlabeled_mask, epoch):
        positive_mask = positive_mask.unsqueeze(dim=0).float()
        unlabeled_mask = unlabeled_mask.unsqueeze(dim=0).float()

        p_weight = torch.pow((torch.ones_like(pred) - torch.clamp(torch.sigmoid(pred), min=0, max=0.999)),
                             self.focal_weight)

        if epoch < self.warm_up_epoch:
            self.loss = 'bce'
        else:
            self.loss = self.saved_loss

        if self.loss == 'sigmoid':
            positive_p_loss = sigmoid_loss(pred) * positive_mask * p_weight
            unlabeled_loss = sigmoid_loss(-pred)
        elif self.loss == 'bce':
            positive_p_loss = bce_loss(pred, positive_mask) * positive_mask * p_weight
            unlabeled_loss = bce_loss(pred, unlabeled_mask, positive=False)
        else:
            NotImplemented

        if self.loss == 'sigmoid':
            positive_n_loss = unlabeled_loss * positive_mask
        elif self.loss == 'bce':
            positive_n_loss = bce_loss(pred, positive_mask, positive=False) * positive_mask
        unlabeled_n_loss = unlabeled_loss * unlabeled_mask

        estimated_p_loss = positive_p_loss.sum() / positive_mask.sum()

        estimated_u_n_loss = unlabeled_n_loss.sum() / unlabeled_mask.sum()
        estimated_p_n_loss = positive_n_loss.sum() / positive_mask.sum()

        estimated_n_loss = torch.abs(estimated_u_n_loss - self.prior * estimated_p_n_loss) / (1 - self.prior)

        loss = self.class_weight * estimated_p_loss + (1 - self.class_weight) * estimated_n_loss

        return loss, estimated_p_loss, estimated_n_loss, estimated_u_n_loss, estimated_p_n_loss
