import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy
import numpy as np
import os
import pytest

from ..log import extract_log_from_text, extract_keys_from_log

def test_extract_keys_from_log():

  # Test with a short log

    log =   {
        "log_filename": "test/test_create_clusterings/log-clustering-2026-07-08--10:01:15.json",
        "config_clustering": {
            "k_range": [
                1,
                25
            ],
            "KMeans": {
                "model_kw": {},
                "fit_predict_kw": {},
                "scaler": "sklearn.preprocessing._data.StandardScaler",
                "scaler_kw": {},
                "model": "sklearn.cluster._kmeans.KMeans"
            },
            "Agglomerative-Single": {
                "model_kw": {
                    "linkage": "single",
                    "metric": "euclidean"
                },
                "fit_predict_kw": {},
                "scaler": None,
                "scaler_kw": {},
                "model": "sklearn.cluster._agglomerative.AgglomerativeClustering"
            },
        }
    }

    keys = extract_keys_from_log(log)
    assert isinstance(keys, dict)
    assert "config_clustering" in keys
    assert "KMeans" in keys["config_clustering"]
    assert "model_kw" in keys["config_clustering"]["KMeans"]
    assert "linkage" in keys["config_clustering"]["Agglomerative-Single"]["model_kw"]
