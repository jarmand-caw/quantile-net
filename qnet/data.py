import copy
import numpy as np
from scipy.stats import yeojohnson
import logging
from sklearn.preprocessing import MinMaxScaler

from qnet import Config

logger = logging.getLogger(__file__)


class Data:
    def __init__(
            self,
            df,
            config=None
    ):
        self.df = df.copy()
        self.df_init = self.df.copy()
        if type(config) == dict:
            self.config = Config(config)
        else:
            self.config = copy.deepcopy(config)

        if type(self.config.target) == str:
            self.config.target = [self.config.target]

        if self.config.yeo_transform is None:
            self.config.yeo_transform = False

        if self.config.yeo_transform:
            self.lambda_dict = {}

        self.category_map_dict = {}
        self.scaler = None

        cols_to_scale = copy.copy(self.config.continuous_features)
        cols_to_scale += self.config.target
        self.cols_to_scale = cols_to_scale

        self._fit()
        self.display_df = None

    def _prep_cat_cols(self):

        if self.config.categorical_features is not None:
            for col in self.config.categorical_features:
                self.df[col] = self.df[col].fillna("nan")
                options = list(dict.fromkeys(list(self.df[col])))
                if "nan" in options:
                    options.remove("nan")
                options.insert(0, "nan")
                integers = [x for x in range(len(options))]
                self.category_map_dict[col] = dict(zip(options, integers))
                self.df[col] = self.df[col].map(self.category_map_dict[col])

        if self.config.multi_label_categorical_features is not None:
            for col in self.config.multi_label_categorical_features:
                total_list = []
                col_values = self.df[col].values
                max_len = 0
                for idx in range(len(self.df)):
                    this_list = col_values[idx]
                    if len(this_list) > max_len:
                        max_len = len(this_list)
                    total_list += this_list

                max_len = max(int(max_len)*1.5, 8)
                logger.info("Max input length for {} feature: {}".format(col, max_len))
                unique_list = list(dict.fromkeys(total_list))
                if "nan" in unique_list:
                    unique_list.remove("nan")
                unique_list.insert(0, "nan")
                integers = [x for x in range(len(unique_list))]
                item_map = dict(zip(unique_list, integers))

                self.category_map_dict[col] = item_map

                for idx in range(len(self.df)):
                    self.df[col].iat[idx] = [item_map[x] for x in self.df[col].iat[idx]]

                array_list = []
                for l in self.df[col].copy().values:
                    leng = len(l)
                    new_l = [0] * int((max_len - leng))
                    new_l = new_l + l
                    array_list.append(np.array(new_l))

                self.df[col] = array_list

    def _prep_cont_cols(self):
        if self.config.continuous_features is not None:
            if self.config.yeo_transform:
                for col in self.config.continuous_features:
                    xt, lmbd = yeojohnson(self.df[col].values)
                    self.df[col] = xt
                    self.lambda_dict[col] = lmbd

        scaler = MinMaxScaler()
        self.df[self.cols_to_scale] = scaler.fit_transform(self.df[self.cols_to_scale])

        self.scaler = scaler

    def transform(self, df):
        df = df.copy()
        if self.config.target[0] not in list(df.columns):
            df[self.config.target[0]] = [0 for x in range(len(df))]

        if self.config.categorical_features is not None:
            for col in self.config.categorical_features:
                df[col] = df[col].fillna("nan")
                df[col] = [
                    self.category_map_dict[col].get(x)
                    if self.category_map_dict[col].get(x) is not None
                    else 0
                    for x in list(df[col])
                ]

        if self.config.multi_label_categorical_features is not None:
            for col in self.config.multi_label_categorical_features:
                for idx in range(len(df)):
                    df[col].iat[idx] = [
                        self.category_map_dict[col].get(x)
                        if self.category_map_dict[col].get(x) is not None
                        else 0
                        for x in df[col].iat[idx]
                    ]
                array_list = []
                max_len = len(self.df[col].iloc[0])
                for l in df[col].copy().values:
                    leng = len(l)
                    new_l = [0] * (max_len-leng)
                    new_l = new_l + l
                    array_list.append(np.array(new_l))
                df[col] = array_list

        if self.config.continuous_features is not None:
            if self.config.yeo_transform:
                for col in self.config.continuous_features:
                    xt = yeojohnson(df[col].values, lmbda=self.lambda_dict[col])
                    df[col] = xt

        df[self.cols_to_scale] = self.scaler.transform(df[self.cols_to_scale])
        return df

    def _fit(self):
        self._prep_cont_cols()
        self._prep_cat_cols()

    def get_inputs(self, input_df=None, transformed=True):
        if input_df is None:
            input_df = self.df.copy()
        else:
            input_df = input_df.copy()

        if not transformed:
            input_df = self.transform(input_df)

        return_dict = {}
        return_dict["cont"] = input_df[self.config.continuous_features].values
        for col in self.config.categorical_features + self.config.multi_label_categorical_features:
            return_dict[col] = input_df[col].values

        return_dict["target"] = input_df[self.config.target].values

        return return_dict

    def create_display_data(
            self, other_cols_to_keep=None, cols_to_drop=None, target_rescale=1
    ):
        other_cols_to_keep = [] if other_cols_to_keep is None else other_cols_to_keep
        cols_to_drop = [] if cols_to_drop is None else cols_to_drop
        assert type(other_cols_to_keep) == list, "other cols to keep must be list. Is type {}".format(type(other_cols_to_keep))
        assert type(cols_to_drop) == list, "cols to drop must be of type list. Is type {}".format(type(cols_to_drop))

        loc_cols = (
            other_cols_to_keep + self.config.continuous_features + self.config.categorical_features + self.config.multi_label_categorical_features
        )
        loc_cols = [col for col in loc_cols if col not in cols_to_drop] + self.config.target
        df_init = self.df_init.copy()
        df_init.reset_index(inplace=True, drop=False)
        display_df = df_init[loc_cols].copy()

        display_df[self.config.target] = display_df[self.config.target] * target_rescale
        cols_to_scale = [x for x in self.cols_to_scale if x in loc_cols]
        display_df[cols_to_scale] = display_df[cols_to_scale].round(2)

        for col in self.config.multi_label_categorical_features:
            col_display = []
            for l in display_df[col]:
                col_display.append(",".join(l))
            display_df[col] = col_display

        self.display_df = display_df

        return display_df


