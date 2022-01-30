import copy
import logging
from itertools import combinations

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import torch
import yaml
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold

from modeling_tools.quantile_net.loss import QuantileLoss
from modeling_tools.quantile_net.nn_utils import (
    generate_loss_sigma,
    yaml_numpy_converter,
)
from modeling_tools.quantile_net.train import Engine
from modeling_tools.xgb_model.xgb_kfold import run_kfold_xgb

logger = logging.getLogger(__file__)


# run-kfold
def run_kfold(
    df,
    train_config,
    model_config,
    num_folds=5,
):
    """
    Runs kfold cross validation on the standard neural network

    Args:
        df: dataframe
        num_folds: number of folds in the kfold

    Returns: the combined targets and outputs from each of the kfolds.

    """
    kfold = KFold(num_folds, shuffle=False)
    df.index = [x for x in range(len(df))]

    total_targets = []
    total_outputs = []
    engine = None
    output_dictionary = dict()
    fold_r2_list = []
    fold_num = 0

    for train_idx, test_idx in kfold.split(df):

        fold_num += 1
        train_df, test_df = df.loc[train_idx], df.loc[test_idx]

        engine = Engine(
            train_config,
            model_config,
            df_init=train_df,
        )

        engine.train()
        target_list, output_array = engine.evaluate(test_df=test_df, for_kfold=True)

        total_targets += target_list
        total_outputs.append(output_array)

        fold_dictionary = dict()
        fold_dictionary["target"] = target_list
        fold_dictionary["output"] = output_array

        fold_r2_list.append(r2_score(target_list, output_array[:, -1]))
        output_dictionary[fold_num] = fold_dictionary

    total_output_tensor = torch.cat(total_outputs)
    sigma = generate_loss_sigma(
        df[model_config["target"]].values, engine.train_config.quantiles
    )
    quantile_crit = QuantileLoss(engine.train_config.quantiles, sigma, device="cpu")
    quantile_loss = quantile_crit(total_output_tensor, torch.Tensor(total_targets))
    r2 = r2_score(total_targets, total_output_tensor[:, -1])

    total_dict = dict()
    total_dict["target"] = total_targets
    total_dict["output"] = total_output_tensor
    output_dictionary["overall"] = total_dict

    logger.info("Loss std by fold: {}".format(np.std(fold_r2_list)))
    logger.info("Quantile Loss: {}".format(quantile_loss))
    logger.info("Overall R2 Score: {}".format(r2))

    return output_dictionary


# What do I really want to do...
# Given a study and an initial config, I want to run 5 kfolds for the top 5 studies and use the results of
# Those kfolds to compare to
class Analysis:
    def __init__(
        self,
        df,
        model_config,
        dict_of_studies,
        dict_of_train_configs,
        num_folds=10,
        number_of_param_combinations=5,
    ):
        assert list(dict_of_studies.keys()) == list(
            dict_of_train_configs.keys()
        ), "dict of studies and of train configs must have same keys"

        self.df = df.copy()
        self.df.index = [x for x in range(len(self.df))]
        self.model_config = copy.deepcopy(model_config)
        self.dict_of_studies = dict_of_studies
        self.dict_of_train_configs = dict_of_train_configs

        self.num_folds = num_folds
        self.number_of_param_combinations = number_of_param_combinations

        self.results_dict = dict()

    def get_list_of_configs(self, study, best_config):
        study_df = study.trials_dataframe().sort_values("value")
        param_col_names = [x for x in study_df.columns if "params" in x]
        param_names = [x.split("params_")[-1] for x in param_col_names]

        config_list = []
        for row_ix in range(self.number_of_param_combinations):
            new_config = copy.deepcopy(best_config)
            for i in range(len(param_names)):
                col = param_col_names[i]
                param = param_names[i]
                if param == "regularization":
                    param = "l2_regularization"
                new_config[param] = study_df.iloc[row_ix][col]
            config_list.append(new_config)

        return config_list

    def run_one_kfold(self, train_config):
        kfold = KFold(self.num_folds, shuffle=False)

        r2_list = []
        rmse_list = []
        quantile_loss_list = []
        overall_target_list = []
        overall_output_list = []
        for train_idx, test_idx in kfold.split(self.df):
            train_df, test_df = self.df.loc[train_idx], self.df.loc[test_idx]

            engine = Engine(
                train_config,
                self.model_config,
                df_init=train_df,
            )

            engine.train()
            target_list, output_array = engine.evaluate(test_df=test_df, for_kfold=True)

            r2_list.append(r2_score(target_list, output_array[:, -1]).item())
            rmse_list.append(
                np.sqrt(mean_squared_error(target_list, output_array[:, -1])).item()
            )

            sigma = generate_loss_sigma(
                self.df[self.model_config["target"]].values,
                engine.train_config.quantiles,
            )
            quantile_crit = QuantileLoss(
                engine.train_config.quantiles, sigma, device="cpu"
            )
            quantile_loss = quantile_crit(output_array, torch.Tensor(target_list))
            quantile_loss_list.append(quantile_loss.item())

            overall_target_list += target_list
            overall_output_list.append(output_array)

        output_tensor = torch.cat(overall_output_list)
        logger.info(output_tensor.shape)
        return (
            r2_list,
            rmse_list,
            quantile_loss_list,
            r2_score(overall_target_list, output_tensor[:, -1]).item(),
        )

    def run_kfold_for_one_study(self, study_name, model_type="nn"):
        study = self.dict_of_studies[study_name]
        initial_config = self.dict_of_train_configs[study_name]
        config_list = self.get_list_of_configs(study, initial_config)

        total_r2_list = []
        total_rmse_list = []
        total_quantile_loss_list = []
        overall_r2_list = []
        for config in config_list:
            if model_type == "nn":
                r2_list, rmse_list, quantile_loss_list, overall_r2 = self.run_one_kfold(
                    config
                )
            elif model_type == "xgb":
                r2_list, rmse_list, quantile_loss_list, overall_r2 = run_kfold_xgb(
                    self.df, self.model_config, config, num_folds=self.num_folds
                )
                overall_r2 = overall_r2.item()
            else:
                assert False, "Model type must be nn or xgb"

            total_r2_list += r2_list
            total_rmse_list += rmse_list
            total_quantile_loss_list += quantile_loss_list
            overall_r2_list.append(overall_r2)

        result_dict = dict()
        result_dict["fold_r2_list"] = total_r2_list
        result_dict["fold_rmse_list"] = total_rmse_list
        result_dict["fold_quantile_loss_list"] = total_quantile_loss_list
        result_dict["overall_r2_list"] = overall_r2_list

        self.results_dict[study_name] = result_dict

        logger.info("{} kfolds completed.".format(study_name))

    def run_kfold_for_all_studies(self, save=True, save_file=None, model_type="nn"):
        study_name_list = list(self.dict_of_studies.keys())
        for study_name in study_name_list:
            self.run_kfold_for_one_study(study_name, model_type=model_type)

        if save:
            yaml.dump(yaml_numpy_converter(self.results_dict), open(save_file, "w"))
            logger.info("Results dict saved successfully.")

    def bengio_modified_t_test(self, score_set_one, score_set_two):
        # https://www.cs.waikato.ac.nz/~eibe/pubs/bouckaert_and_frank.pdf
        # A student's t-test requires independence, which is broken in this case
        # This test aims to correct for that

        k = self.num_folds
        r = self.number_of_param_combinations

        # Compute the difference between the results
        diff = [y - x for y, x in zip(score_set_one, score_set_two)]

        n1 = int(len(self.df) * 0.9)
        # compute the number of data points used for testing
        n2 = int(len(self.df) * 0.1)
        # compute the variance of differences
        sigma2 = np.var(diff)

        numerator = (1 / (k * r)) * np.sum(diff)
        denominator = np.sqrt(((1 / (k * r)) * (n2 / n1)) * sigma2)

        t_statistic = numerator / denominator

        p_value = (1 - stats.t.cdf(np.abs(t_statistic), k * r - 1)) * 2.0

        return p_value

    def create_p_value_grid(self):
        grid = np.ones((len(self.dict_of_studies), len(self.dict_of_studies)))
        grid_df = pd.DataFrame(grid)
        grid_df.columns = list(self.dict_of_studies.keys())
        grid_df.index = list(self.dict_of_studies.keys())
        grid_df.replace(1, np.nan, inplace=True)

        unique_pairs = combinations(list(self.dict_of_studies.keys()), 2)
        for x, y in unique_pairs:
            score_set_x = self.results_dict[x]["fold_r2_list"]
            score_set_y = self.results_dict[y]["fold_r2_list"]
            p_val = self.bengio_modified_t_test(score_set_x, score_set_y)
            grid_df.at[x, y] = p_val
            grid_df.at[y, x] = p_val

        return grid_df


def create_p_value_grid(results_dict, k, r, n_datapoints, metric="fold_r2_list"):
    """

    Args:
        results_dict: dictionary of results with keys being identifiers for different models
        k: number of folds in k fold
        r: number of repititions of k fold
        n_datapoints: number of datapoints

    Returns: dataframe giving the pairwise p-values of the bengio modified t test between models

    """
    grid = np.ones((len(results_dict), len(results_dict)))
    grid_df = pd.DataFrame(grid)
    grid_df.columns = list(results_dict.keys())
    grid_df.index = list(results_dict.keys())
    grid_df.replace(1, np.nan, inplace=True)

    n_training = int((1 - (1 / k)) * n_datapoints)
    n_testing = int((1 / k) * n_datapoints)

    unique_pairs = combinations(list(results_dict.keys()), 2)
    for x, y in unique_pairs:
        score_set_x = results_dict[x][metric]
        score_set_y = results_dict[y][metric]
        p_val = bengio_modified_t_test(
            score_set_x, score_set_y, k, r, n_training, n_testing
        )
        grid_df.at[x, y] = p_val
        grid_df.at[y, x] = p_val

    return grid_df


def bengio_modified_t_test(score_set_one, score_set_two, k, r, n_training, n_testing):
    # https://www.cs.waikato.ac.nz/~eibe/pubs/bouckaert_and_frank.pdf
    # A student's t-test requires independence, which is broken in this case
    # This test aims to correct for that

    # Compute the difference between the results
    diff = [y - x for y, x in zip(score_set_one, score_set_two)]

    # compute the variance of differences
    sigma2 = np.var(diff)

    numerator = (1 / (k * r)) * np.sum(diff)
    denominator = np.sqrt(((1 / (k * r)) * (n_testing / n_training)) * sigma2)

    t_statistic = numerator / denominator

    p_value = (1 - stats.t.cdf(np.abs(t_statistic), k * r - 1)) * 2.0

    return p_value