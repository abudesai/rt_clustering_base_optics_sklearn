from dataclasses import replace
import dis
import numpy as np, pandas as pd
import joblib
import sys
import os, warnings
from sklearn.cluster import OPTICS

warnings.filterwarnings("ignore")


model_fname = "model.save"

MODEL_NAME = "clustering_base_optics"


class ClusteringModel:
    def __init__(
        self,
        min_samples,
        metric="minkowski",
        cluster_method="xi",
        min_cluster_size=0.1,
        **kwargs,
    ) -> None:
        # def __init__(self, min_samples, metric="cosine", cluster_method="xi", min_cluster_size=0.4, **kwargs) -> None:
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_method = cluster_method
        self.min_cluster_size = min_cluster_size

        self.model = self.build_model()

    def build_model(self):
        model = OPTICS(
            min_samples=self.min_samples,
            metric=self.metric,
            cluster_method=self.cluster_method,
            min_cluster_size=self.min_cluster_size,
        )
        return model

    def fit_predict(self, *args, **kwargs):
        return self.model.fit_predict(*args, **kwargs)

    def evaluate(self, x_test):
        """Evaluate the model and return the loss and metrics"""
        raise NotImplementedError

    def save(self, model_path):
        joblib.dump(self, os.path.join(model_path, model_fname))

    @classmethod
    def load(cls, model_path):
        clusterer = joblib.load(os.path.join(model_path, model_fname))
        return clusterer


def save_model(model, model_path):
    model.save(model_path)


def load_model(model_path):
    model = ClusteringModel.load(model_path)
    return model


def get_data_based_model_params(data):
    """
    Set any model parameters that are data dependent.
    For example, number of layers or neurons in a neural network as a function of data shape.
    """
    # min_samples = min(3 * data.shape[1] + 1, max(1, int(0.03 * data.shape[0])), 10)
    min_samples = int(max(5, 0.03 * data.shape[0]))
    return {"min_samples": min_samples}
