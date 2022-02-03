from datetime import timedelta
from logging import getLogger

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


logger = getLogger(__file__)


def use_optimizer(network, config):
    """
    Convenience function to make loading an optimizer easier
    Args:
        network: pytorch neural net module
        config: hyperparmeters specific to the optimizer

    Returns: The pytorch optimizer object
    """
    if config.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.l2_regularization,
            nesterov=True,
        )
        return optimizer
    elif config.optimizer == "adam":
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=config.lr,
            weight_decay=config.l2_regularization,
        )
        return optimizer
    else:
        raise NotImplementedError(
            "optimizer not implemented. Options are 'sgd' and 'adam'"
        )


class EmbedDataset(Dataset):
    """
    Extension of the pytorch dataset class to handle the satellite lists where there are multiple satellites that
    require a bagged embedding
    """

    def __init__(self, input_dict):
        self.input_dict = input_dict

    def __len__(self):
        return len(list(self.input_dict.values())[0])

    def __getitem__(self, idx):
        output_dict = {}
        for key, values in self.input_dict.items():
            output_dict[key] = values[idx]

        return output_dict


def convert_to_date_int(df, date_string):

    date = pd.to_datetime(date_string)
    row_date_int = df.iloc[0]["date int"]
    row_date = df.iloc[0]["first pop day"]
    row_date = pd.to_datetime(row_date, unit="ms")
    ref_day = row_date - timedelta(row_date_int)
    date_int = date - ref_day

    return date_int.days


def yaml_numpy_converter(d):
    for key, value in d.items():
        if (type(value) == np.int64) or (type(value) == np.float64):
            d[key] = value.item()
    return d