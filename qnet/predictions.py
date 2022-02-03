import logging
import os

import joblib
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader

from qnet.probability import generate_cdf_pdf_plot
from qnet.models import QuantileNet
from qnet.nn_utils import EmbedDataset

logger = logging.getLogger(__file__)


class Predictions(object):

    """
    The Predictions class handles everything related to using your trained model to make predictions.
    The core functionality predicts the value for a point, as well as the value of each quantile
    for that point the model was trained to predict. The other crucial piece of functionality finds
    the data points most similar to your datapoint. Additional features include generating PDF and CDF graphs
    from the predicted quantiles, as well as converting the predictions and similar points to easy to read
    dataframes
    """

    model_config_file_name = "model_config.yaml"
    train_config_file_name = "train_config.yaml"
    model_file_name = "model.pth"
    data_transformer_file_name = "data_transformer.sav"
    embeddings_file_name = "embedded_data.sav"

    def __init__(self, dir):
        """
        Args:
            dir:
        """
        self.average_days_in_a_month = 30.4

        self.data_transformer = joblib.load(
            os.path.join(dir, self.data_transformer_file_name)
        )
        self.model = QuantileNet.load(dir, map_location="cpu")
        self.model.eval()
        if os.path.exists(os.path.join(dir, self.embeddings_file_name)):
            self.existing_embeddings = joblib.load(
                os.path.join(dir, self.embeddings_file_name)
            )
        else:
            self.existing_embeddings = None

        # Specific to each prediction. Will be reset when a new prediction is made
        self.kwarg_input = None
        self.kwarg_output = None
        self.pop = None
        self.bandwidth = None

        self.predictions = None

        self.predictions_per_month = None
        self.total_cost_predictions = None

        self.embedding = None
        self.similarity = None
        self.sorted_similarities = None
        self.sorted_similarity_idx = None

        self.cdf = None
        self.pdf = None
        self.confidence = None
        self.low_mhz_per_day = None
        self.low_mhz_per_month = None
        self.low_total = None

        self.high_mhz_per_day = None
        self.high_mhz_per_month = None
        self.high_total = None
        self.pred_df = None
        self.similar_df = None

    def reset_prediction_parameters(self):

        self.input_df = None
        self.kwarg_output = None
        self.predictions = None
        self.predictions_per_month = None
        self.total_cost_predictions = None
        self.embedding = None
        self.similarity = None
        self.sorted_similarities = None
        self.sorted_similarity_idx = None
        self.cdf = None
        self.pdf = None
        self.confidence = None
        self.low_mhz_per_day = None
        self.low_mhz_per_month = None
        self.low_total = None
        self.high_mhz_per_day = None
        self.high_mhz_per_month = None
        self.high_total = None
        self.pred_df = None
        self.similar_df = None

    def predict(self, input_df):
        self.reset_prediction_parameters()
        self.input_df = input_df
        kwarg_output = self.data_transformer.get_inputs(
            input_df=input_df, transformed=False
        )
        self.kwarg_output = kwarg_output
        dataset = EmbedDataset(self.kwarg_output)
        dataloader = DataLoader(dataset, batch_size=len(dataset))
        predictions = None
        for batch in dataloader:
            predictions = self.model(**batch).view(-1).detach().numpy()

        blank_array = np.zeros(
            (len(predictions), len(self.data_transformer.cols_to_scale))
        )
        # The last column of the array should contain the predictions
        blank_array[:, -1] = predictions
        predictions = self.data_transformer.scaler.inverse_transform(blank_array)[:, -1]
        self.predictions = predictions

        return predictions

    def get_monthly_predictions(self):
        predictions_per_month = self.predictions * self.average_days_in_a_month
        self.predictions_per_month = predictions_per_month
        return predictions_per_month

    def get_total_cost_predictions(self, pop, bandwidth):
        self.pop = pop
        self.bandwidth = bandwidth
        total_cost_predictions = self.predictions * self.pop * self.bandwidth
        self.total_cost_predictions = total_cost_predictions
        return total_cost_predictions

    def generate_embedding(self, input_df=None):

        inputs = self.data_transformer.get_inputs(input_df, transformed=False)
        dataset = EmbedDataset(inputs)
        dataloader = DataLoader(dataset)
        combined_vector = None
        for batch in dataloader:
            combined_vector = self.model(return_embeddings=True, **batch)

        self.embedding = combined_vector
        return combined_vector

    def find_most_similar_points(
        self,
        input_df=None,
    ):
        if input_df is None:
            input_df = self.input_df.copy()

        self.generate_embedding(input_df)

        self.similarity = cosine_similarity(
            self.embedding.to("cpu"), self.existing_embeddings.to("cpu")
        ).view(-1)
        sorted_similarities, sorted_similarity_idx = torch.sort(
            self.similarity, descending=True
        )

        self.sorted_similarities, self.sorted_similarity_idx = (
            sorted_similarities,
            sorted_similarity_idx,
        )

        return sorted_similarities, sorted_similarity_idx

    def create_pdf_cdf(self):

        if self.predictions_per_month is None:
            raise ValueError("Must run self.get_monthly_predictions first")

        self.cdf, self.pdf = generate_cdf_pdf(
            self.model.config.quantiles, self.predictions_per_month[:-1]
        )

        return self.cdf, self.pdf

    def get_confidence_range(self, confidence):

        self.confidence = confidence

        upper_quantile = round(1 - (1 - float(confidence)) / 2, 4)
        lower_quantile = round((1 - float(confidence)) / 2, 4)
        lower_index = self.model.config.quantiles.index(lower_quantile)
        upper_index = self.model.config.quantiles.index(upper_quantile)

        self.low_mhz_per_day = self.predictions[lower_index]
        self.low_mhz_per_month = self.predictions_per_month[lower_index]
        self.low_total = self.low_mhz_per_day * self.pop * self.bandwidth

        self.high_mhz_per_day = self.predictions[upper_index]
        self.high_mhz_per_month = self.predictions_per_month[upper_index]
        self.high_total = self.high_mhz_per_day * self.bandwidth * self.pop

        return {
            "low-per-day": self.low_mhz_per_day,
            "low-per-month": self.low_mhz_per_month,
            "low-total": self.low_total,
            "high-per-day": self.high_mhz_per_day,
            "high-per-month": self.high_mhz_per_month,
            "high-total": self.high_total,
        }

    def prediction_todf(self):
        pred_df = pd.DataFrame(
            {
                "": ["Total Cost", "Cost per MHz per Month"],
                "Predicted": [
                    f"${round(self.total_cost_predictions[-1].item(), 0):,.0f}",
                    f"${round(self.predictions_per_month[-1].item(), 0):,.0f}",
                ],
                "Low {}% Prediction Interval".format(self.confidence): [
                    f"${round(self.low_total, 0):,.0f}",
                    f"${round(self.low_mhz_per_month, 0):,.0f}",
                ],
                "High {}% Prediction Interval".format(self.confidence): [
                    f"${round(self.high_total, 0):,.0f}",
                    f"${round(self.high_mhz_per_month, 0):,.0f}",
                ],
            }
        )
        self.pred_df = pred_df
        return pred_df

    def create_similar_df(self):
        similar_df = self.data_transformer.create_display_data(
            other_cols_to_keep=["index", "first pop day"],
            cols_to_drop=["date int"],
            target_rescale=self.average_days_in_a_month,
        )
        similar_df = similar_df.loc[self.sorted_similarity_idx]
        similar_df.reset_index(inplace=True, drop=False)
        similar_df["similarity"] = self.sorted_similarities.detach().numpy()
        self.similar_df = similar_df
        return similar_df
