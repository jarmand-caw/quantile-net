import logging
import os
from datetime import datetime

import joblib
import optuna
import pandas as pd
import yaml
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from qnet.nn_utils import EmbedDataset
from qnet.train import Engine

logger = logging.getLogger(__file__)


class OptunaStudy(object):
    def __init__(
            self,
            df,
            model_config,
            base_config,
            save_folder,
            n_trials=100,
            optimize_model_architecture=False,
            optimize_batch_size=False,
            optimize_momentum=True,
            optimize_mse=True,
    ):
        """

        Args:
            df:
            model_config:
            base_config:
            save_folder:
            n_trials:
            optimize_model_architecture:
            optimize_batch_size:
            optimize_momentum:
            optimize_mse:
        """
        self.model_config = model_config
        self.base_config = base_config
        self.n_trials = n_trials
        self.save_folder = save_folder

        self.df = df
        self.train_df, self.test_df = train_test_split(
            self.df, test_size=0.15, shuffle=False
        )
        self.train_df, self.test_df2 = train_test_split(
            self.train_df, test_size=0.1, shuffle=True, random_state=42
        )
        self.test_df = pd.concat([self.test_df, self.test_df2])

        self.optimize_model_architecture = optimize_model_architecture
        self.optimize_batch_size = optimize_batch_size
        self.optimize_mse = optimize_mse
        self.optimize_momentum = optimize_momentum

    def create_objective_function(self):
        def objective(trial):
            """

            Args:
                trial:

            Returns:

            """
            config = self.base_config.copy()
            if self.optimize_model_architecture:
                embedding_size = trial.suggest_int("embedding_size", 5, 50)
                config["embedding_dims"] = [embedding_size] * (
                        len(self.model_config["categorical_features"])
                        + len(self.model_config["multi_label_categorical_features"])
                )
                cont_width = trial.suggest_int("cont_width", 75, 700)
                cont_depth = trial.suggest_int("cont_depth", 3, 7)
                config["continuous_layers"] = [cont_width] * cont_depth
                cat_width = trial.suggest_int("cat_width", 75, 700)
                cat_depth = trial.suggest_int("cat_depth", 3, 7)
                config["categorical_layers"] = [cat_width] * cat_depth
                final_width = trial.suggest_int("final_width", 75, 700)
                final_depth = trial.suggest_int("final_depth", 3, 7)
                config["final_layers"] = [final_width] * final_depth

            config["lr"] = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
            config["l2_regularization"] = trial.suggest_float(
                "l2_regularization", 1e-1, 0.8
            )
            config["dropout"] = trial.suggest_float("dropout", 0.05, 0.75)
            if config["optimizer"] == "sgd":
                if self.optimize_momentum:
                    config["momentum"] = trial.suggest_float("momentum", 0.5, 0.95)
            if self.optimize_batch_size:
                config["batch_size"] = trial.suggest_float("batch_size", 0.05, 0.5)


            engine = Engine(config, self.model_config, df_init=self.train_df)
            engine.train()

            test_df = engine.data_transformer.transform(self.test_df)
            test_dataset = EmbedDataset(engine.data_transformer.get_inputs(test_df))
            test_loader = DataLoader(
                test_dataset, batch_size=len(test_dataset), num_workers=0
            )

            if self.optimize_mse:
                target, output = engine.evaluate(
                    test_loader=test_loader, for_kfold=True
                )
                testing_score = mean_squared_error(target, output[:, -1])
            else:
                logger.info("Returning raw loss, not mse.")
                testing_score = engine.evaluate(test_loader=test_loader)

            if pd.isnull(testing_score):
                testing_score = 1e8

            return testing_score

        return objective

    def create_total_best_config(self, best_params):
        config = self.base_config.copy()
        if self.optimize_model_architecture:
            embedding_size = best_params["embedding_size"]
            config["embedding_dims"] = [embedding_size] * (
                    len(self.model_config["categorical_features"])
                    + len(self.model_config["multi_label_categorical_features"])
            )
            cont_width = best_params["cont_width"]
            cont_depth = best_params["cont_depth"]
            config["continuous_layers"] = [cont_width] * cont_depth
            cat_width = best_params["cat_width"]
            cat_depth = best_params["cat_depth"]
            config["categorical_layers"] = [cat_width] * cat_depth
            final_width = best_params["final_width"]
            final_depth = best_params["final_depth"]
            config["final_layers"] = [final_width] * final_depth

        config["lr"] = best_params["lr"]
        config["l2_regularization"] = best_params["l2_regularization"]
        config["dropout"] = best_params["dropout"]
        if config["optimizer"] == "sgd":
            if self.optimize_momentum:
                config["momentum"] = best_params["momentum"]
        if self.optimize_batch_size:
            config["batch_size"] = best_params["batch_size"]

        return config

    def run_study(self, existing_study=None):

        if existing_study is not None:
            if type(existing_study) == optuna.Study:
                study = existing_study
            else:
                study = joblib.load(existing_study)
        else:
            study = optuna.create_study(direction="minimize")

        objective = self.create_objective_function()

        try:
            study.optimize(objective, n_trials=self.n_trials)
        except KeyboardInterrupt:
            pass

        now = datetime.now()  # current date and time

        month = now.strftime("%m")
        day = now.strftime("%d")
        hours = now.strftime("%H")
        minutes = now.strftime("%M")
        param_file_name = "train_config_{}_{}_{}_{}.yaml".format(
            month, day, hours, minutes
        )
        study_file_name = "optuna_study_{}_{}_{}_{}.sav".format(
            month, day, hours, minutes
        )

        best_config = self.create_total_best_config(study.best_params)
        yaml.dump(
            best_config, open(os.path.join(self.save_folder, param_file_name), "w")
        )
        joblib.dump(study, open(os.path.join(self.save_folder, study_file_name), "wb"))

        return os.path.join(self.save_folder, param_file_name), os.path.join(
            self.save_folder, study_file_name
        )


if __name__ == "__main__":
    from argparse import ArgumentParser
    from qnet.log_init import initialize_logger

    initialize_logger()
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--data_path",
        type=str,
        help="path to data file",
        required=True
    )
    parser.add_argument(
        "-mc",
        "--model_config_path",
        type=str,
        help="path to model config file",
        required=True,
    ),
    parser.add_argument(
        "-bc",
        "--base_config_path",
        type=str,
        help="path to base config file",
        required=True,
    )
    parser.add_argument(
        "-sd",
        "--save_directory",
        type=str,
        help="path to directory where study and best config will be saved",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--n_trials",
        default=100,
        type=int,
        help="number of study trials"
    )
    parser.add_argument(
        "-oma",
        "--optimize_model_architecture",
        type=bool,
        default=False,
        help="determine if you want the model architecture to be optimized along with hyperparameters",
    )
    parser.add_argument(
        "-obs",
        "--optimize_bs",
        default=True,
        type=bool,
        help="determines if batch size is a paramter to be optimized. defaults to 64",
    )
    parser.add_argument(
        "-omom",
        "--optimize_momentum",
        type=bool,
        default=True,
        help="if using sgd optimizer, determine if you want momentum to be optimized",
    )
    parser.add_argument(
        "-omse",
        "--optimize_mse",
        default=True,
        type=bool,
        help="whether to optimize the mean squared error loss or the output loss",
    )
    parser.add_argument(
        "-exs",
        "--existing_study",
        default=None,
        type=str,
        help="path to the existing study, if you'd like to use one"
    )

    args = parser.parse_args()

    if args.data_path.split(".")[-1] == "json":
        df = pd.read_json(args.data_path)
    elif args.data_path.split(".")[-1] == "csv":
        df = pd.read_csv(args.data_path)
    else:
        raise NotImplementedError("Data type not implemented. Must be csv or json. {}".format(args.data_path.split(".")[-1]))

    base_config = yaml.full_load(open(args.base_config_path))
    model_config = yaml.full_load(open(args.model_config_path))
    study = OptunaStudy(
        df=df,
        model_config=model_config,
        base_config=base_config,
        save_folder=args.save_directory,
        n_trials=args.n_trials,
        optimize_model_architecture=args.optimize_model_architecture,
        optimize_batch_size=args.optimize_bs,
        optimize_momentum=args.optimize_momentum,
        optimize_mse=args.optimize_mse,
    )

    study.run_study(args.existing_study)