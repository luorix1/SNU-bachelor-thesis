import torch


class CircularLoss(torch.nn.Module):
    """
    Custom loss function for gait cycle phase estimation.
    """
    def __init__(self):
        super(CircularLoss, self).__init__()

    def forward(self, y_pred, y_true):
        diff = torch.abs(y_pred - y_true)
        return torch.mean(torch.min(diff, 1 - diff))