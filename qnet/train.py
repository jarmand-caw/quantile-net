import copy
import logging
import os
from datetime import datetime

import joblib
import optuna.exceptions
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from qnet import Config
from qnet.data import Data
from qnet.loss import MultiTaskWrapper
from qnet.models import QuantileNet
from qnet.utils import EmbedDataset, use_optimizer

logger = logging.getLogger(__file__)


class Engine(object):

    model_config_file_name = "model_config.yaml"
    train_config_file_name = "train_config.yaml"
    model_file_name = "model.pth"
    data_transformer_file_name = "data_transformer.sav"
    embeddings_file_name = "embedded_data.sav"
    model_config_keys = [
        "categorical_features",
        "multi_label_categorical_features",
        "continuous_features",
        "target",
        "yeo_transform",
    ]

    def __init__(
        self,
        train_config,
        model_config,
        load_from_checkpoint=False,
        df_init=None,
        data_transformer=None,
        trained_model=None,
    ):
        if type(train_config) == Config:
            self.train_config = copy.deepcopy(train_config)
        else:
            train_config = Config(train_config)
            self.train_config = copy.deepcopy(train_config)
        if type(model_config) == Config:
            self.model_config = copy.deepcopy(model_config)
        else:
            model_config = Config(model_config)
            self.model_config = copy.deepcopy(model_config)

        if load_from_checkpoint:
            assert df_init is None, "if initializing from transformer, cannot give a df"
            assert (datatransformer is not None) & (
                trained_model is not None
            ), "if initializing from transformer, must add the transformer"

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        self.train_config.device = self.device

        if not load_from_checkpoint:
            self.data_transformer = Data(df=df_init, config=self.model_config)
        else:
            self.data_transformer = data_transformer


        if not load_from_checkpoint:
            num_embeddings = []
            for col in (
                self.data_transformer.cat_cols + self.data_transformer.bagged_cat_cols
            ):
                num_embeddings.append(len(self.data_transformer.category_map_dict[col]))

            self.train_config.num_embeddings = num_embeddings
            self.train_config.num_continuous_features = len(
                self.data_transformer.cont_cols
            )

            if self.train_config.embedding_dims is None:
                embedding_dims = []
                for num_embed in self.train_config.num_embeddings:
                    embedding_dims.append(min(50, (num_embed + 1) // 2))
                self.train_config.embedding_dims = embedding_dims

            if self.train_config.quantiles == "full":
                quantiles = (
                    [0.005, 0.025]
                    + [round(0.05 * x, 3) for x in range(1, 20)]
                    + [0.975, 0.995]
                )
                self.train_config.quantiles = quantiles
            elif self.train_config.quantiles == "tail":
                quantiles = [0.005, 0.025, 0.05, 0.1, 0.9, 0.95, 0.975, 0.995]
                self.train_config.quantiles = quantiles
            elif self.train_config.quantiles == "every 0.1":
                quantiles = [x * 0.1 for x in range(1, 10)]
                self.train_config.quantiles = quantiles
            elif self.train_config.quantiles is None:
                raise ValueError("must have an entry for quantiles in config file")
            elif type(self.train_config.quantiles) == list:
                pass
            else:
                raise NotImplementedError("value for quantiles not implemented")


            if self.train_config.train_shuffle_method is None:
                self.train_config.train_shuffle_method = "fixed"
            elif self.train_config.train_shuffle_method not in [
                "fixed",
                "variable",
            ]:
                raise NotImplementedError(
                    "train_shuffle_method not implemented. options are fixed or variable"
                )
            if self.train_config.shuffle_val is None:
                self.train_config.shuffle_val = False

            if self.train_config.val_size is None:
                self.train_config.val_size = 0.1

        self.val_size = self.train_config.val_size
        self.shuffle_val = self.train_config.shuffle_val
        self.train_shuffle_method = self.train_config.train_shuffle_method

        train_df, val_df = train_test_split(
            self.data_transformer.df,
            test_size=self.val_size,
            shuffle=self.shuffle_val,
            random_state=42,
        )

        self.train_config.batch_size_int = int(round(self.train_config.batch_size*len(train_df), 0))
        if self.train_shuffle_method == "fixed":
            train_df = shuffle(train_df, random_state=42)
            train_dataset = EmbedDataset(self.data_transformer.get_inputs(train_df))

            if (len(train_dataset) % self.train_config.batch_size_int) < 20:
                drop_last = True
            else:
                drop_last = False

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_config.batch_size_int,
                shuffle=False,
                num_workers=0,
                drop_last=drop_last,
            )

        elif self.train_shuffle_method == "variable":
            train_dataset = EmbedDataset(self.data_transformer.get_inputs(train_df))
            if (len(train_dataset) % self.train_config.batch_size_int) < 20:
                drop_last = True
            else:
                drop_last = False
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_config.batch_size_int,
                shuffle=True,
                num_workers=0,
                drop_last=drop_last,
            )
        else:
            raise NotImplementedError(
                "train shuffle method not implemented. options are fixed or variable"
            )

        val_dataset = EmbedDataset(self.data_transformer.get_inputs(val_df))
        self.val_loader = DataLoader(
            val_dataset, batch_size=len(val_dataset), num_workers=0
        )

        if load_from_checkpoint:
            trained_model.device = self.device
            self.mtl = MultiTaskWrapper(
                trained_model, self.train_config.quantiles, self.device
            )
            self.opt = use_optimizer(self.mtl, self.train_config)
        else:
            self.mtl = MultiTaskWrapper(
                QuantileNet(Config(self.get_model_building_config())),
                self.train_config.quantiles,
                self.device,
            )

            self.opt = use_optimizer(self.mtl, self.train_config)

        self.verbose = self.train_config.verbose
        self.num_epochs = self.train_config.num_epochs
        self.patience = self.train_config.patience
        self.count = 0
        self.use_scheduler = self.train_config.use_scheduler
        if self.use_scheduler:
            self.scheduler = ReduceLROnPlateau(
                self.opt, "min", patience=3, factor=0.2, verbose=True, cooldown=4
            )
            self.epochs_before_scheduler = self.train_config.epochs_before_scheduler

        self.best_loss = 1e10
        self.best_model = None

        self.model_config_file_path = None
        self.scaler_file_path = None
        self.category_map_path = None
        self.model_path = None
        if self.model_config.yeo_transform:
            self.lambda_dict_path = None
        self.data_transformer_path = None
        self.existing_embeddings = None

    def train_single_batch(self, kwargs):
        self.opt.zero_grad()
        if self._mtl_flag:
            loss = self.mtl(kwargs)
        else:
            pred = self.model(**kwargs)
            loss = self.crit(pred, kwargs["target"].float().view(-1).to(self.device))
        loss.backward()
        self.opt.step()
        loss_item = loss.item()
        return loss_item

    def _set_model_mode(self, mode):
        if mode == "train":
            if self._mtl_flag:
                self.mtl.model.train()
            else:
                self.model.train()
        elif mode == "eval":
            if self._mtl_flag:
                self.mtl.model.eval()
            else:
                self.model.eval()
        else:
            raise NotImplementedError(
                "options for positional argument mode are 'train' or 'eval'"
            )

    def train_an_epoch(self, epoch_id):
        self._set_model_mode("train")
        total_loss = 0
        for batch_id, batch in enumerate(self.train_loader):
            loss_item = self.train_single_batch(batch)
            total_loss += loss_item
        if self.verbose:
            logger.info(
                "Epoch {}, total training loss: {}".format(
                    epoch_id, round(total_loss, 4)
                )
            )

    def evaluate(self, test_df=None, test_loader=None, epoch_id=None, for_kfold=False):
        self._set_model_mode("eval")
        if test_df is not None:
            assert (
                test_loader is None
            ), "Can either use a test_df or test_loader but not both"
            test_df = self.data_transformer.transform(test_df)
            test_dataset = EmbedDataset(self.data_transformer.get_inputs(test_df))
            test_loader = DataLoader(
                test_dataset, batch_size=len(test_dataset), num_workers=0
            )
        with torch.no_grad():
            total_loss = 0
            all_preds = torch.tensor([])
            all_y = []
            if test_loader is None:
                test_loader = self.val_loader
            for i, batch in enumerate(test_loader):
                if for_kfold:
                    if self._mtl_flag:
                        pred = self.mtl.model(**batch).to("cpu")
                        all_preds = torch.cat((all_preds, pred))
                        all_y += list(batch["target"].float().view(-1).to("cpu"))
                    else:
                        pred = self.model(**batch)
                        all_preds = torch.cat((all_preds, pred.to("cpu")))
                        all_y += list(batch["target"].float().view(-1).to("cpu"))
                else:
                    if self._mtl_flag:
                        loss = self.mtl(batch)
                        total_loss += loss.item()
                    else:
                        pred = self.model(**batch)
                        loss = self.crit(
                            pred, batch["target"].float().view(-1).to(self.device)
                        ).item()
                        total_loss += loss

            if for_kfold:
                return all_y, all_preds

            if self.verbose:
                logger.info(
                    "Epoch {}, Testing R2 Score: {}".format(
                        epoch_id, round(total_loss, 4)
                    )
                )
            return total_loss

    def train(self, optuna_prune=False, trial=None):
        for epoch in range(self.num_epochs):
            self.train_an_epoch(epoch)
            loss = self.evaluate(epoch_id=epoch)

            # If using optuna study, this will significantly reduce total study time
            if optuna_prune:
                assert trial is not None, "if using optuna_prune, must provide trial"
                if epoch >= 30:
                    trial.report(loss, epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            if self.use_scheduler:
                if epoch >= self.epochs_before_scheduler:
                    self.scheduler.step(loss)
            if loss < self.best_loss:
                self.best_loss = copy.deepcopy(loss)
                if self._mtl_flag:
                    self.best_model = copy.deepcopy(self.mtl.model.state_dict())
                else:
                    self.best_model = copy.deepcopy(self.model.state_dict())
                self.count = 0
            else:
                self.count += 1
            if self.count >= self.patience:
                logger.info("Model training stopped after {} epochs.".format(epoch))
                logger.info("Best testing loss: {}".format(round(self.best_loss, 4)))
                if self.best_model is None:
                    pass
                else:
                    if self._mtl_flag:
                        self.mtl.model.load_state_dict(self.best_model)
                    else:
                        self.model.load_state_dict(self.best_model)
                break

    def get_model_building_config(self):
        model_keys = [
            "seed",
            "quantiles",
            "dropout",
            "continuous_features",
            "categorical_features",
            "multi_label_categorical_features",
            "num_embeddings",
            "embedding_dims",
            "num_continuous_features",
            "continuous_layers",
            "categorical_layers",
            "final_layers",
            "target",
            "yeo_transform",
            "device",
        ]
        config = {
            model_key: getattr(
                Config({**vars(self.model_config), **vars(self.train_config)}),
                model_key,
            )
            for model_key in model_keys
        }
        return config

    def generate_embeddings(self):
        inputs = self.data_transformer.get_inputs()
        dataset = EmbedDataset(inputs)
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        self._set_model_mode("eval")

        combined_vector = None
        if self._mtl_flag:
            for batch in dataloader:
                combined_vector = self.mtl.model(return_embeddings=True, **batch)
        else:
            for batch in dataloader:
                combined_vector = self.model(return_embeddings=True, **batch)

        self.existing_embeddings = combined_vector.to("cpu")

    def save(self, dir=None, append_auto_dir=True, save_embeddings=False):
        if append_auto_dir:
            now = datetime.now()  # current date and time
            month = now.strftime("%m")
            day = now.strftime("%d")
            hours = now.strftime("%H")
            minutes = now.strftime("%M")
            append = "model_{}_{}_{}_{}".format(month, day, hours, minutes)
            dir = os.path.join(dir, append)
        if os.path.exists(dir) and (os.listdir(dir)):
            logger.warning("Directory is not empty. Overwriting")
            for f in os.listdir(dir):
                os.remove(os.path.join(dir, f))
        os.makedirs(dir, exist_ok=True)

        self.model_path = os.path.join(dir, self.model_file_name)
        self.model_config_file_path = os.path.join(dir, self.model_config_file_name)
        self.train_config_file_path = os.path.join(dir, self.train_config_file_name)

        self.data_transformer_path = os.path.join(dir, self.data_transformer_file_name)
        model_config = self.get_model_building_config()
        model_config["device"] = None
        yaml.dump(model_config, open(self.model_config_file_path, "w"))
        yaml.dump(
            {**vars(self.train_config), **vars(self.model_config)},
            open(self.train_config_file_path, "w"),
        )
        if self._mtl_flag:
            torch.save(
                self.mtl.model.state_dict(),
                self.model_path,
            )
        else:
            torch.save(self.model.state_dict(), self.model_path)
        joblib.dump(self.data_transformer, open(self.data_transformer_path, "wb"))
        if save_embeddings:
            self.embeddings_path = os.path.join(dir, self.embeddings_file_name)
            if self.existing_embeddings is None:
                self.generate_embeddings()
            joblib.dump(self.existing_embeddings, open(self.embeddings_path, "wb"))

        return dir

    @classmethod
    def load(cls, dir):
        train_config_file_path = os.path.join(
            dir, getattr(cls, "train_config_file_name")
        )
        train_config = yaml.unsafe_load(open(train_config_file_path))
        model_config = {
            k: v
            for k, v in train_config.items()
            if k in getattr(cls, "model_config_keys")
        }
        train_config = {
            k: v for k, v in train_config.items() if k not in list(model_config.keys())
        }
        model = QuantileNet.load(dir)

        data_transformer_path = os.path.join(
            dir, getattr(cls, "data_transformer_file_name")
        )
        data_transformer = joblib.load(open(data_transformer_path, "rb"))

        engine = cls(
            train_config,
            model_config,
            load_from_checkpoint=True,
            datatransformer=data_transformer,
            trained_model=model,
        )

        return engine


if __name__ == "__main__":
    from argparse import ArgumentParser
    from modeling_tools.log_init import initialize_logger

    initialize_logger()
    parser = ArgumentParser(description="Train a model to go into production")

    parser.add_argument(
        "-d", "--data_path", type=str, help="path to data file", required=True
    )
    parser.add_argument(
        "-tc",
        "--train_config_path",
        type=str,
        help="path to neural net train config file",
        required=True,
    )
    parser.add_argument(
        "-mc",
        "--model_config_path",
        type=str,
        help="path to the base model config file",
        required=True,
    )
    parser.add_argument(
        "-sd",
        "--save_dir",
        type=str,
        help="path to folder for saving model",
        required=True,
    )

    args = parser.parse_args()
    if args.data_path.split(".")[-1] == "json":
        df = pd.read_json(args.data_path)
    elif args.data_path.split(".")[-1] == "csv":
        df = pd.read_csv(args.data_path)
    else:
        raise NotImplementedError(
            "Data type not implemented. Must be csv or json. {}".format(args.data_path.split(".")[-1]))
    train_config = yaml.full_load(open(args.train_config_path))
    model_config = yaml.full_load(open(args.model_config_path))
    engine = Engine(train_config, model_config, df_init=df)
    engine.train()
    dir = engine.save(args.save_dir, save_embeddings=True)
    logger.info("Successfully saved at {}".format(dir))