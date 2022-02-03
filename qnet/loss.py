import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__file__)


class MultiTaskWrapper(nn.Module):
    def __init__(self, model, quantiles, device):
        super(MultiTaskWrapper, self).__init__()
        self.model = model
        self.model.to(device)
        self.device = device
        self.quantiles = quantiles
        self.num_tasks = len(self.quantiles) + 1
        self.log_vars = nn.Parameter(torch.zeros((self.num_tasks)))
        self.mse_crit = nn.MSELoss()

    def forward(self, kwargs):
        out = self.model(**kwargs)
        pred = out[:, -1]
        quantile_pred = out[:, :-1]

        target = kwargs["target"].float().view(-1).to(self.device)
        mse_loss = self.mse_crit(pred, target)
        adjusted_mse_loss = (
                mse_loss / (torch.exp(2 * self.log_vars[-1])) + self.log_vars[-1]
        )

        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - quantile_pred[:, i]
            q_l = torch.max((q - 1) * errors, q * errors).unsqueeze(1)
            q_l_adjusted = q_l / (torch.exp(2 * self.log_vars[i])) + self.log_vars[i]
            losses.append(q_l_adjusted)

        batch_by_quantiles = torch.cat(losses, dim=1)
        quantile_loss = torch.mean(torch.sum(batch_by_quantiles, dim=1))

        total_loss = adjusted_mse_loss + quantile_loss

        return total_loss
