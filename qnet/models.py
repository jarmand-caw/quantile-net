import copy
import logging
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml

from qnet import Config

logger = logging.getLogger(__file__)


class QuantileNet(nn.Module):

    model_config_file_name = "model_config.yaml"
    model_file_name = "model.pth"

    def __init__(self, config):
        super(QuantileNet, self).__init__()

        if type(config) == dict:
            self.config = Config(copy.copy(config))
        else:
            self.config = copy.copy(config)

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)

        if hasattr(self.config, "device"):
            self.device = self.config.device
        else:
            self.device = "cpu"

        logger.info("Model on device: {}".format(self.device))

        self.num_quantiles = len(self.config.quantiles)

        self.embedding_dict = nn.ModuleDict({})
        for i, cat_feature in enumerate(self.config.categorical_features):
            self.embedding_dict[cat_feature] = nn.Embedding(
                num_embeddings=self.config.num_embeddings[i],
                embedding_dim=self.config.embedding_dims[i],
                padding_idx=0
            )

        for i, bag_cat_feature in enumerate(self.config.multi_label_categorical_features):
            i = i+len(self.config.categorical_features)
            self.embedding_dict[bag_cat_feature] = nn.EmbeddingBag(
                num_embeddings=self.config.num_embeddings[i],
                embedding_dim=self.config.embedding_dims[i],
                padding_idx=0,
                mode="mean"
            )

        self.continuous_layer_sizes = self.config.continuous_layers
        self.continuous_layer_sizes.insert(0, self.config.num_continuous_features)

        if (self.config.categorical_features is not None) or (
            self.config.multi_label_categorical_features is not None
        ):
            self.total_embedding_dim = 0
            for x in self.config.embedding_dims:
                self.total_embedding_dim += x
            self.categorical_layer_sizes = self.config.categorical_layers
            self.categorical_layer_sizes.insert(0, self.total_embedding_dim)

        if not ((self.config.categorical_features is None) and (self.config.multi_label_categorical_features is None)):
            if self.num_cont>0:
                self.first_bottom_size = (
                    self.categorical_layer_sizes[-1] + self.continuous_layer_sizes[-1]
                )
            else:
                self.first_bottom_size = (
                    self.categorical_layer_sizes[-1]
                )

        self.final_layer_sizes = self.config.final_layers
        self.final_layer_sizes.insert(0, self.first_bottom_size)

        if self.num_cont > 0:
            self.cont_layers = nn.ModuleList()
            for idx in range(len(self.continuous_layer_sizes) - 1):
                in_size = self.continuous_layer_sizes[idx]
                out_size = self.continuous_layer_sizes[idx + 1]
                self.cont_layers.append(nn.Linear(in_size, out_size))
                self.cont_layers.append(nn.ReLU())
                self.cont_layers.append(nn.BatchNorm1d(out_size))
                self.cont_layers.append(nn.Dropout(self.config.dropout))
            self.cont_mlp = nn.Sequential(*self.cont_layers)

            # The embed mlp
        if (len(self.config.categorical_features) + len(self.config.multi_label_categorical_features)) > 0:
            self.categorical_layers = nn.ModuleList()
            for idx in range(len(self.categorical_layer_sizes) - 1):
                in_size = self.categorical_layer_sizes[idx]
                out_size = self.categorical_layer_sizes[idx + 1]
                self.categorical_layers.append(nn.Linear(in_size, out_size))
                self.categorical_layers.append(nn.ReLU())
                self.categorical_layers.append(nn.BatchNorm1d(out_size))
                self.categorical_layers.append(nn.Dropout(self.config.dropout))
            self.categorical_mlp = nn.Sequential(*self.categorical_layers)

            # The final mlp
        self.final_layers = nn.ModuleList()
        out_size = None
        for idx in range(len(self.final_layer_sizes) - 1):
            in_size = self.final_layer_sizes[idx]
            out_size = self.final_layer_sizes[idx + 1]
            self.final_layers.append(nn.Linear(in_size, out_size))
            self.final_layers.append(nn.ReLU())
            self.final_layers.append(nn.BatchNorm1d(out_size))
        self.final_layers.append(nn.Linear(out_size, self.num_quantiles + 1))
        self.final_mlp = nn.Sequential(*self.final_layers)

    def forward(self, return_embeddings=False, **kwargs):
        cat_vector = None
        cont_output = None
        cat_output = None

        if self.num_cont > 0:
            cont = kwargs.pop("cont")
            cont = cont.float().to(self.device)
            batch_size = cont.shape[0]
            cont_output = self.cont_mlp(cont)
        else:
            batch_size = list(kwargs.values())[0].shape[0]

        if (len(self.config.categorical_features) + len(self.config.multi_label_categorical_features)) > 0:
            embeddings = []
            for key in self.config.categorical_features + self.config.multi_label_categorical_features:
                values = kwargs[key]
                values = values.long().to(self.device)
                emb = self.embedding_dict[key]
                if emb is None:
                    raise ValueError(
                        "the keyword arguments must match the categorical columns in config"
                    )
                embed = emb(values).view(batch_size, -1)
                embeddings.append(embed)

            cat_vector = torch.cat(
                embeddings,
                dim=-1,
            )
            cat_output = self.categorical_mlp(cat_vector)

        if return_embeddings:
            return torch.cat([cat_vector, cont_output], dim=-1)

        if self.num_cont == 0:
            combined_vector = cat_output
        elif (len(self.config.categorical_features) + len(self.config.multi_label_categorical_features)) == 0:
            combined_vector = cont_output
        else:
            combined_vector = torch.cat([cat_output, cont_output], dim=-1)

        out = self.final_mlp(combined_vector)
        return out

    @classmethod
    def load(cls, dir, map_location=None):
        model_config_save_path = os.path.join(dir, "model_config.yaml")
        model_save_path = os.path.join(dir, "model.pth")
        model_config = yaml.full_load(open(model_config_save_path, "r"))
        model = cls(model_config)

        if map_location is None:
            if torch.cuda.is_available():
                logger.info("Loading device: {}".format("cuda"))
                device = torch.device("cuda")
            else:
                logger.info("Loading device : {}".format("cpu"))
                device = torch.device("cpu")
        else:
            device = torch.device(map_location)

        model.load_state_dict(torch.load(model_save_path, map_location=device))

        return model
