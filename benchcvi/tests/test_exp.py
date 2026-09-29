
import numpy as np
import pytest
from unittest.mock import patch

from benchcvi.utils import (write_json)
from benchcvi.log import extract_log_from_text

from benchcvi.exp import (
    create_clusterings, prepare_data, compute_CVI_values, re_use_experiment
)

config_clustering_barton = {
    "config_clustering" : {
        "KMeans" : {
            "k_range" : [1, 5],
            "model" : "sklearn.cluster.KMeans",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw" : {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "Agglomerative-Single" : {
            "k_range" : [1, 5],
            "model": "sklearn.cluster.AgglomerativeClustering",
            "model_kw": {
                "linkage": "single",
                "metric": "euclidean"
            },
            "fit_predict_kw": {},
            "scaler_kw": {}
        },
        "Agglomerative-Ward" : {
            "k_range" : [1, 5],
            "model": "sklearn.cluster.AgglomerativeClustering",
            "model_kw": {
                "linkage": "ward",
                "metric": "euclidean"
            },
            "fit_predict_kw": {},
            "scaler": None,
            "scaler_kw": {}
        },
        "KMedoids" : {
            "k_range" : [1, 5],
            "model": "kmedoids.KMedoids",
            "model_kw": {
                "metric": "euclidean",
                "random_state" : 221,
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
    }
}

config_clustering_UCR = {
    "config_clustering" : {
        "TimeSeriesKMeans" : {
            "k_range" : [1, 4],
            "model": "aeon.clustering.TimeSeriesKMeans",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "TimeSeriesKMedoids" : {
            "k_range" : [1, 4],
            "model": "aeon.clustering.TimeSeriesKMedoids",
            "model_kw" : {
                "random_state" : 221
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        },
        "Agglomerative-Single" : {
            "k_range" : [1, 4],
            "model": "sklearn.cluster.AgglomerativeClustering",
            "model_kw": {
                "linkage": "single",
                "metric": "pycvi.dist.time_series_metric_with_sklearn"
            },
            "fit_predict_kw": {},
            "scaler": "sklearn.preprocessing.StandardScaler",
            "scaler_kw": {}
        }
    }
}

config_data_CVI = {
    "config_data": {
        "path_data" : "example_data/",
        "path_res" : "test/test_prepare_data/",
        "max_n_samples": 1000,
        "max_n_labels": 10,
        "max_n_dims": 10,
        "exclude": ["arrhythmia", "2d-4c-no"],
        "include_only": ["artificial/"]
    },
    "config_CVI": {
        "seed": 221,
        "Hartigan": {
            "cvi": "pycvi.cvi.Hartigan",
            "cvi_kw" : {
                "rng" : 221
            }
        },
        "Inertia-sum": {
            "cvi": "pycvi.cvi.Inertia",
            "cvi_kw": {
                "reduction": "sum"
        }
        },
        "Diameter-max": {
            "cvi": "pycvi.cvi.Diameter",
            "cvi_kw": {
                "reduction": "max"
            }
        }
    },
}

config1 = {
    **config_data_CVI,
    **config_clustering_barton,
}

config1_UCR = {
    **config_data_CVI,
    **config_clustering_UCR,
}


def test_re_use_experiment():
    # An input without a CVI or clustering log should be rejected.
    with pytest.raises(ValueError, match="must be a CVI log or a clustering log"):
        re_use_experiment({})

    # A missing experiment file means there is nothing to reuse.
    log_exp = {"log_clustering": {"log_fname": "missing.json"}}
    with patch("benchcvi.exp.os.path.isfile", return_value=False):
        assert re_use_experiment(log_exp) is False

    # A consistent clustering log can be reused.
    log_exp = {"log_clustering": {"log_fname": "existing.json"}}
    with (
        patch("benchcvi.exp.os.path.isfile", return_value=True),
        patch("benchcvi.exp.interpret_dict", return_value={}),
        patch("benchcvi.exp.consistent_log_exp", return_value=True),
    ):
        assert re_use_experiment(log_exp) is True

    # A consistent CVI log can also be reused.
    log_exp = {"log_CVI": {"log_fname": "existing.json"}}
    with (
        patch("benchcvi.exp.os.path.isfile", return_value=True),
        patch("benchcvi.exp.interpret_dict", return_value={}),
        patch("benchcvi.exp.consistent_log_exp", return_value=True),
    ):
        assert re_use_experiment(log_exp) is True

    # re_run forces a consistent log to be recomputed instead of reused.
    log_exp = {"log_clustering": {"log_fname": "existing.json"}}
    with (
        patch("benchcvi.exp.os.path.isfile", return_value=True),
        patch("benchcvi.exp.interpret_dict", return_value={}),
        patch("benchcvi.exp.consistent_log_exp", return_value=True),
    ):
        assert re_use_experiment(log_exp, re_run=True) is False

    # An inconsistent existing log raises unless overwrite or re_run is set.
    log_exp = {"log_clustering": {"log_fname": "existing.json"}}
    with (
        patch("benchcvi.exp.os.path.isfile", return_value=True),
        patch("benchcvi.exp.interpret_dict", return_value={}),
        patch("benchcvi.exp.consistent_log_exp", return_value=False),
    ):
        with pytest.raises(FileExistsError):
            re_use_experiment(log_exp)

    # overwrite allows replacement of an inconsistent log without reusing it.
    log_exp = {"log_clustering": {"log_fname": "existing.json"}}
    with (
        patch("benchcvi.exp.os.path.isfile", return_value=True),
        patch("benchcvi.exp.interpret_dict", return_value={}),
        patch("benchcvi.exp.consistent_log_exp", return_value=False),
    ):
        assert re_use_experiment(log_exp, overwrite=True) is False

    # re_run also allows replacement of an inconsistent log.
    log_exp = {"log_clustering": {"log_fname": "existing.json"}}
    with (
        patch("benchcvi.exp.os.path.isfile", return_value=True),
        patch("benchcvi.exp.interpret_dict", return_value={}),
        patch("benchcvi.exp.consistent_log_exp", return_value=False),
    ):
        assert re_use_experiment(log_exp, re_run=True) is False


def test_prepare_data():

    dir = "test/test_prepare_data"
    config_fname = f"{dir}/config.json"
    write_json(config_fname, config1)

    log = prepare_data(config_fname)

    assert isinstance(log, dict)
    assert "config_data" in log
    assert "log_data" in log
    assert log["config_data"] == config1["config_data"]

    # Make sure that the text log file contain 2 logs and that the last log corresponds to the log dictionary
    l_log_extracted = extract_log_from_text(f"{log['log_data']['log_fname']}.txt")

    assert isinstance(l_log_extracted, list)
    assert len(l_log_extracted) == 2
    assert l_log_extracted[-1] == log

def test_create_clusterings():

    dir = "test/test_create_clusterings"
    config_fname = f"{dir}/config.json"
    config2 = config1.copy()
    config2["config_data"]["path_res"] = f"{dir}/"
    write_json(config_fname, config2)

    log = create_clusterings(config_fname,re_run=True)

    assert isinstance(log, dict)
    assert "config_data" in log
    assert "log_data" in log
    assert "log_clustering" in log
    assert "config_clustering" in log
    assert log["config_data"] == config2["config_data"]

    # Make sure that the text log file contain 2 logs and that the last log corresponds to the log dictionary
    l_log_extracted = extract_log_from_text(f"{log['log_clustering']['log_fname']}.txt")

    assert isinstance(l_log_extracted, list)
    assert len(l_log_extracted) == 2
    assert l_log_extracted[-1] == log

def test_compute_CVI_values():
    dir = "test/test_compute_CVI_values"

    # Preparing folder and config

    config_fname_barton = f"{dir}/config-barton.json"
    config_barton = config1.copy()

    config_fname_ucr = f"{dir}/config-ucr.json"
    config_ucr = config1_UCR.copy()

    configs = [config_barton, config_ucr]
    config_fnames = [config_fname_barton, config_fname_ucr]

    for config_fname, config in zip(config_fnames, configs):
        config["config_data"]["path_res"] = f"{dir}/"
        write_json(config_fname, config)

        # ------- Test when re-computing everything --------
        log = compute_CVI_values(config_fname, re_run=True)

        assert isinstance(log, dict)
        assert "config_data" in log
        assert "log_data" in log
        assert "log_clustering" in log
        assert "config_clustering" in log
        assert "log_CVI" in log
        assert "config_CVI" in log
        assert log["config_data"] == config["config_data"]

        # Make sure that the text log file contain 2 logs and that the last log corresponds to the log dictionary
        l_log_extracted = extract_log_from_text(f"{log['log_CVI']['log_fname']}.txt")

        assert isinstance(l_log_extracted, list)
        assert len(l_log_extracted) == 2
        assert l_log_extracted[-1] == log


        # ------- Test when starting from previous log --------
        log_exp = log["log_clustering"]["log_fname"]
        log_bis = compute_CVI_values(config_fname, log_exp, re_run=True)

        # Check that everything that isn't the log_fname or time is the same
        log_bis_copy = log_bis.copy()
        log_copy = log.copy()

        del log_copy["log_filename"]
        del log_bis_copy["log_filename"]

        del log_bis_copy["log_CVI"]["log_fname"]
        del log_copy["log_CVI"]["log_fname"]

        del log_bis_copy["log_CVI"]["time"]
        del log_copy["log_CVI"]["time"]

        assert log_bis_copy == log_copy

