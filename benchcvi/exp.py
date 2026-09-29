"""Experiment orchestration for full pipeline."""


import os
import sys
import argparse
from datetime import datetime
import numpy as np
import time
from pathlib import Path

from pycvi.cluster import get_clustering, generate_all_clusterings
from pycvi.compute_scores import compute_all_scores
from pycvi.exceptions import SelectionError
from pycvi.dist import time_series_metric_with_sklearn

from .config import interpret_config, get_exp_config, get_mandatory_keys
from .data import (
    all_summary_stats, find_datasets, filter_datasets, load_data_labels, is_time_series,
    all_summary_stats
)
from .log import (
    save_log, print_log, consistent_log_data, consistent_log_clustering,
    consistent_log_exp
)
from .utils import interpret_dict
from .clustering import (
    compute_VI_quality, filter_experiments, group_exp_by_dataset
)

from typing import Union

def prepare_data(config_fname:str) -> dict:
    """
    Prepare datasets and write a data-selection log.

    - Reads the config file: function `interpret_config`
    - Prints the initiated log file containing the config file used `print_log` function
    - Find all datasets in the path_data folder (using the function `find_datasets`)
    - filter datasets based on the path_data and constraints defined in the config file (using the function `filter_datasets`)
      - Creates a list of kept datasets
      - Creates the dictionary of sorted (kept, and one key per constraint) datasets
    - Creates a json resulting logfile `log-data-20XX-XX-XX.json`, with
      - One key `config_data` with the corresponding dict of the data config file used
      - One key `log_data` with the corresponding dict:
          - `dropped_datasets` a dict of lists of dataset names as ``[full/path/to/DATASET]`` (see `filter_datasets` function)
          - `kept_datasets` list of dataset names as ``[full/path/to/DATASET]`` (see `filter_datasets` function)
          - `log_fname` : `path/to/res/log-data-20XX-XX-XX` (without the `.txt` or `.json`)
    - Save the merged dictionary ``log-data-20XX-XX-XX.json``  with the function `save_log`
    - Save output log file ``log-data-20XX-XX-XX.txt``
    - Returns the merged dictionary ``log-data-20XX-XX-XX.json`` as a dict

    Parameters
    ----------
    config_fname : str
        Path to the configuration file containing at least a
        ``config_data`` section.

    Returns
    -------
    dict
        Log dictionary containing the interpreted data config, dataset
        filtering results, and the generated log filename.
    """
    # ---------------- Read config file ---------------------
    config = interpret_config(config_fname)
    if "config_data" not in config:
        raise ValueError("The config file must contain a 'config_data' key.")
    path_data = config['config_data']['path_data']
    path_res = config['config_data']['path_res']

    # ----------- Prepare current log files ------------------
    full_date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log_fname = f'{path_res}log-data-{full_date}'

    p = Path(log_fname)
    p.parent.mkdir(parents=True, exist_ok=True)

    fout = open(f"{log_fname}.txt", 'wt')
    sys.stdout = fout

    log = {
        "config_data": config['config_data'],
        "log_filename" : None,
        "log_data": {
            "log_fname": log_fname,
        }}

    # Make the log visible in the output file
    print_log(log)

    # ---------------- Find datasets ------------------------
    datasets = find_datasets(path_data)

    # --------------- Filter datasets ----------------------
    constraints = config['config_data'].copy()
    constraints.pop('path_data', None)
    constraints.pop('path_res', None)

    filtered_datasets = filter_datasets(
        datasets, path_data=path_data, **constraints
    )

    # ----- Summary stats for sanity checks --------------
    stats = all_summary_stats(path_data, filtered_datasets["kept_datasets"])

    # -------------------- Finalize log and save --------------
    log['log_data'].update(filtered_datasets)
    log["log_data"].update(stats)

    save_log(
        f"{log_fname}.json", log,
        overwrite=True, add_date=False, new_name=True, verbose=False,
    )

    # Make the log visible in the output file
    print_log(log)

    fout.close()
    return log

def re_use_experiment(
    log_exp,
    re_run:bool = False,
    overwrite:bool = False,
) -> bool:
    """
    Check if an experiment can be reused based on its log file.

    Parameters
    ----------
    log_exp : dict
        Log dictionary of the experiment to check.
    re_run : bool, optional
        If True, the experiment will be re-run regardless of its current status.
    overwrite : bool, optional
        If True, the existing log file will be overwritten if it is not consistent with the current experiment.

    Returns
    -------
    bool
        True if the experiment can be reused, False otherwise.

    Raises
    ------
    FileExistsError
        If a file with the expected log filename already exists and cannot be overwritten.
    """

    # Check if it is a CVI exp or a clustering exp
    if "log_CVI" in log_exp:
        log_exp_fname = log_exp["log_CVI"]["log_fname"]
    elif "log_clustering" in log_exp:
        log_exp_fname = log_exp["log_clustering"]["log_fname"]
    else:
        raise ValueError("Experiment log must be a CVI log or a clustering log to be reused.")

    re_use = False

    # Check if a file already exists with the expected log filename
    if os.path.isfile(log_exp_fname):
        print(f"A file {log_exp_fname} already exists")

        # Load the existing file
        existing_exp_log = interpret_dict(log_exp_fname)

        # Check if it is consistent with current experiment expectations
        is_consistent = consistent_log_exp(
            log_exp, existing_exp_log
        )
        if is_consistent:
            print(f"{log_exp_fname} is consistent with current data file status and clustering configuration." )
        else:
            print(f"{log_exp_fname} is NOT consistent with current data file status and clustering configuration." )

        if re_run:
            print(f"Overwriting {log_exp_fname}, because `re_run` is set to True.")
        elif overwrite and not is_consistent:
            print(f"Overwriting {log_exp_fname}, because `overwrite` is set to True and the existing log is not consistent with current data file status and clustering configuration.")
        elif is_consistent:
            print(f"Re-using existing {log_exp_fname}, because the existing log is consistent with current data file status and clustering configuration.")
            re_use = True
        else:
            raise FileExistsError(f"A file {log_exp_fname} already exists. \n 1. Set `re_run` to True to overwrite it regardless of its content \n 2. Set `overwrite` to True to overwrite it if the existing log is not consistent with current data file status and clustering configuration.\n 3.Use a different experience name or a different path_res to avoid overwriting the existing file.")

    return re_use

def create_clusterings(
    config_fname:str,
    log_data_fname:Union[str, None] = None,
    re_run:bool = False,
    overwrite:bool = False,
) -> dict:
    """
    Create clustering experiments for each dataset in the log file

    - Extract clustering experiment configuration from the config file
    - Get list of kept datasets based on log_data
    - for each dataset
        - Load data and labels
        - Get the true clusters from the labels
        - Decide whether whether to use ts_dist based on data shape
        for each clustering experiment in the config file:
            - instantiate scaler
            - call pycvi generate all clusterings
                - data
                - a model class
                - n_clusters_range
                - ts_dist (based on data shape)
                - model_kw
                - fit_predict_kw
                - an instantiated scaler (should be instantiated beforehand)
                - verbose
            - save clustering file
                - clusterings
                - VI
                - quality

    Parameters
    ----------
    config_fname : str
        Path to the configuration file containing at least a
        ``config_clustering`` section.
    log_data_fname : Union[str, None], optional
        Path to an existing data log. When omitted, :func:`prepare_data`
        is called first.
    re_run : bool, optional
        If True, the experiment will be re-run regardless of its current status.
    overwrite : bool, optional
        If True, the experiment will be overwritten if the existing log is not consistent with the current data file status and clustering configuration.

    Returns
    -------
    dict
        Log dictionary combining data-selection information, clustering
        configuration, experiment file paths, and timing information.

    Raises
    ------
    FileExistsError
        If a file with the expected experiment log filename already
        exists and cannot be overwritten.
    InconsistentLogError
        If the existing data log is not consistent with the current data
        file status.
    """
    # ------------- Read config file and previous log ------------------
    t_start = time.time()

    config = interpret_config(config_fname)
    if "config_clustering" not in config:
        raise ValueError("The config file must contain a 'config_clustering' key.")

    if log_data_fname is None:
        log_data = prepare_data(config_fname)
    else:
        log_data = interpret_dict(log_data_fname)
        # Check that the log is consistent with the current state
        # - all kept_datasets exist
        # - summary stats of log_data consistent with data file status
        consistent_log_data(log_data)

    path_data = log_data['config_data']['path_data']
    path_res = log_data['config_data']['path_res']

    # ----------- Prepare current log files ---------------------
    full_date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log_fname = f'{path_res}log-clustering-{full_date}'

    p = Path(log_fname)
    p.parent.mkdir(parents=True, exist_ok=True)

    fout = open(f"{log_fname}.txt", 'wt')
    sys.stdout = fout

    log = {
        **log_data,
        "config_clustering": config['config_clustering'],
        "log_filename" : None,
        "log_clustering": {
            "log_fname": log_fname,
            "path_exp": [],
            "exp_names" : [],
        }}

    # Make the log visible in the output file
    print_log(log)

    # ================ Create clustering experiments =====================

    # Get the clustering experiments configuration from the config file
    exps_config = get_exp_config(config)
    k_range = range(*config["config_clustering"]["k_range"])

    # ------------------ Load datasets ------------------------
    path_datasets = log_data['log_data']['kept_datasets']

    for d in path_datasets:

        # Load dataset, labels and true clustering
        data, labels = load_data_labels(f"{path_data}{d}")
        if len(data.shape) == 2:
            (N, D) = data.shape
            T = 1
        elif len(data.shape) == 3:
            (N, T, D) = data.shape
        clustering_true = get_clustering(labels)
        k_ref = len(np.unique(labels))

        print(f" ======== DATASET {d} | k_ref = {k_ref} ========== ")

        data, ts_dist = is_time_series(data)

        # Prepare main log to store VI and quality for each exp
        log["log_clustering"][d] = {}

        for exp_name, exp_config in exps_config["config_clustering"].items():

            print(f" ------------- Experience {exp_name} ---------------- ")
            t_start_exp = time.time()

            # ----------- Prepare clustering log --------------

            log_exp_fname = f"{path_res}{exp_name}/{d}-clustering.json"

            # Fill what we can in the log exp (to then do consistency checks)
            log_exp = {
                "info_dataset": {
                    "path_data": path_data,
                    "dataset_name": d,
                    d: log["log_data"][d], # add the summary statistics
                },
                "config_clustering" : {
                    "exp_name" : exp_name,
                    exp_name: exp_config,  # Add the config of this experiment
                },
                "log_clustering": {
                    "log_fname" : log_exp_fname,
                    "ts_dist" : ts_dist,
                },
            }

            # Check if we should re_use
            re_use = re_use_experiment(
                log_exp, re_run=re_run, overwrite=overwrite
            )

            if re_use:
                # load already saved experiment
                log_exp = interpret_dict(log_exp_fname)

                VIs = log_exp["log_clustering"]["VIs"]
                qualities = log_exp["log_clustering"]["qualities"]
                k_q_best = log_exp["log_clustering"]["k_q_best"]
                dt = log_exp["log_clustering"]["time"]

            else:

                # Prepare main log to store VI and quality for each exp
                log["log_clustering"][d][exp_name] = {}

                # ----------- Generate all clusterings --------------

                # Instantiate a scaler if one was provided in the config file
                if exp_config['scaler'] is None:
                    scaler = None
                else:
                    scaler = exp_config['scaler'](**exp_config['scaler_kw'])

                # Special case if pycvi.dist.time_series_metric_with_sklearn
                # is used somewhere in model_kw, we need to provide d
                model_kw = exp_config['model_kw'].copy()
                for k, v in model_kw.items():
                    if v == time_series_metric_with_sklearn:
                        model_kw[k] = time_series_metric_with_sklearn(d=D)

                # Generate all clusterings for the current dataset and exp
                clusterings = generate_all_clusterings(
                    data=data,
                    model_class=exp_config['model'],
                    n_clusters_range=k_range,
                    ts_dist=ts_dist,
                    scaler=scaler,
                    model_kw=model_kw,
                    fit_predict_kw=exp_config['fit_predict_kw'],
                    verbose=1,
                )

                # ----------- Compute VI and quality --------------
                VIs, qualities = compute_VI_quality(clustering_true, clusterings)

                # Find k that maximise quality
                k_q_best = max(qualities, key=qualities.get)

                # ----------- Finalize log and save --------------
                t_end_exp = time.time()
                dt = float(f"{t_end_exp - t_start_exp:.2f}")
                print(f"\n\nExperiment done in: {dt:.2f}s")

                # --- Update log exp ---
                log_exp["log_clustering"]["clusterings"] = clusterings
                log_exp["log_clustering"]["VIs"] = VIs
                log_exp["log_clustering"]["qualities"] = qualities
                log_exp["log_clustering"]["k_q_best"] = k_q_best
                log_exp["log_clustering"]["time"] = dt

                save_log(
                    f"{log_exp_fname}", log_exp,
                    overwrite=True, add_date=False, new_name=True, verbose=False,
                )

            # --- Update main log ---
            # Add current experiment log filename to the main log file
            log["log_clustering"]["path_exp"].append(log_exp_fname)
            # Add current experiment name
            log["log_clustering"]["exp_names"].append(exp_name)

            log["log_clustering"][d][exp_name]["VIs"] = VIs
            log["log_clustering"][d][exp_name]["qualities"] = qualities
            log["log_clustering"][d][exp_name]["k_q_best"] = k_q_best
            log["log_clustering"][d][exp_name]["time"] = dt


    # -------------------- Finalize log and save -----------------------
    t_end = time.time()
    dt = float(f"{t_end - t_start:.2f}")
    print(f"\n\nTotal execution time: {dt:.2f}s")

    log['log_clustering']["time"] = dt
    save_log(
        f"{log_fname}.json", log,
        overwrite=True, add_date=False, new_name=True, verbose=False,
    )
    # Make the log visible in the output file
    print_log(log)

    fout.close()
    return log

def compute_CVI_values(
    config_fname:str,
    log_clustering_fname:Union[str, None] = None,
    re_run:bool = False,
    overwrite:bool = False,
) -> dict:
    """
    Create CVI result files for each non-filtered clustering experiment.

    - Relying on `log-clustering-20XX-XX-XX.json`.
    - Creates a text logfile `log-CVI-20XX-XX-XX.txt`. Fully reads `config-data.json` and `config-clustering.json`, `config-CVI.json` at the very beginning of the logfile.
    - Creates a json logfile `log-CVI-20XX-XX-XX.json` concatenating the config files used and giving some info about the general results
    - One key `config_data` with the corresponding dict
    - One key `log_data` with the corresponding dict
    - One key `config_clustering` with the corresponding dict
    - One key `log_clustering` with the corresponding dict
    - One key `config_CVI` with the provided config
    - One key `log_CVI` with the following keys:
        - `path_CVI` = list of all CVI json files created `[path/to/res/clustering_name/path/to/dataset-CVI.json]`
        - One key `kept_experiments` a list `"path/to/experiment/dataset-clustering.json"`, see `filter_experiments` function
        - One key `dropped_experiments` a dict `contraint : "path/to/experiment/dataset-clustering.json"`, see `filter_experiments` function
        - One key `kept_datasets` a list `[path/to/dataset]` for which at least one clustering method was kept
        - One key `dropped_datasets` a list `[path/to/dataset]` for which no clustering method was kept
        - `CVI_names` = list of CVI names used
        - `log_fname` : `path/to/res/log-CVI-20XX-XX-XX` (without the `.txt` or `.json`)
        - `time`
    - Filter experiments based on the contraints defined in `config_CVI` (see `filter_experiments` function)
    - Compute the values for each kept experiment and each provided CVI
    - Creates a `path_res/clustering_method/dataset-CVI.json` file for each kept experiment
    - `dataset`
    - `k_ref`
    - `exp_name`
    - `main_log_fname` (CVI)
    - `log_filename` (-CVI)
    - `log_experiment`
        - `main_log_fname` (clustering)
        - `log_fname` (-clustering)
        - exp_name (experiment config)
        - `ts_dist`
    - `CVI_names`
    - CVI_name
        - `CVI_values`
        - `k_selected`
        - `time`

    Parameters
    ----------
    config_fname : str
        Path to the configuration file containing at least a
        ``config_CVI`` section.
    log_clustering_fname : Union[str, None], optional
        Path to an existing clustering log. When omitted,
        :func:`create_clusterings` is called first.
    re_run : bool, optional
        If True, the experiment will be re-run regardless of its current status.
    overwrite : bool, optional
        If True, the experiment will be overwritten if the existing log is not consistent with the current data file status and clustering configuration.

    Returns
    -------
    dict
        Log dictionary containing CVI configuration, filtered
        experiments, generated CVI file paths, and timing information.

    Raises
    ------
    FileExistsError
        If a file with the expected experiment log filename already
        exists and cannot be overwritten.
    InconsistentLogError
        If the existing data log or clustering experiment log og
        clustering log are not consistent with the current data file
        status.
    """
    # ------------- Read config file and previous log ------------------
    t_start = time.time()

    config = interpret_config(config_fname)
    if "config_CVI" not in config:
        raise ValueError("The config file must contain a 'config_CVI' key.")

    if log_clustering_fname is None:
        log_clustering = create_clusterings(
            config_fname, re_run=re_run, overwrite=overwrite,
        )
    else:
        log_clustering = interpret_dict(log_clustering_fname)
        # Check that the log is consistent with the current state
        # - all kept_datasets exist
        # - summary stats of log_data consistent with data file status
        consistent_log_data(log_clustering)
        # - all path_exp exist
        # - summary stats of log_exp consistent with log_data
        consistent_log_clustering(log_clustering)

    path_data = log_clustering['config_data']['path_data']
    path_res = log_clustering['config_data']['path_res']


    # Get the clustering experiments configuration from the config file
    exps_config = get_exp_config(config)
    CVI_names = list(exps_config["config_CVI"].keys())

    # ----------- Prepare current log files ---------------------
    full_date = datetime.today().strftime('%Y-%m-%d--%H-%M-%S')
    log_fname = f'{path_res}log-CVI-{full_date}'

    p = Path(log_fname)
    p.parent.mkdir(parents=True, exist_ok=True)

    fout = open(f"{log_fname}.txt", 'wt')
    sys.stdout = fout

    log = {
        **log_clustering,
        "config_CVI": config['config_CVI'],
        "log_filename" : None,
        "log_CVI": {
            "log_fname": log_fname,
            "CVI_names": CVI_names,
            "path_CVI_files" : [],
        }}

    # Make the log visible in the output file
    print_log(log)

    # ================ Find experiments ===================
    all_experiments = log_clustering['log_clustering']['path_exp']

    # ================ Filter experiments ===================

    # Find constraints from config by finding mandatory keys != seed
    constraint_names = [
        k for k in get_mandatory_keys()['config_CVI']["mandatory"]
        if k != "seed"
    ]
    constraints = {
        c: config['config_CVI'][c] for c in constraint_names
        if c in config['config_CVI']
    }

    # Filter experiments and datasets
    filtered_exp, filtered_datasets = filter_experiments(
        all_experiments, **constraints
    )

    # Group experiments by dataset
    grouped_exp = group_exp_by_dataset(filtered_exp["kept_experiments"])

    # Note that we could avoid grouping by dataset but then it's bit less clean
    # and we would have to load the data and labels for each experiment
    for d, experiments in grouped_exp.items():

        data, labels = load_data_labels(f"{path_data}{d}")
        k_ref = len(np.unique(labels))

        print(f" ======== DATASET {d} | k_ref = {k_ref} ========== ")

        data, ts_dist = is_time_series(data)

        # Prepare main log to store the selected k for each CVI and dataset
        log["log_CVI"][d] = {"k_ref": k_ref}

        for exp in experiments:

            # ----------- Read clustering log --------------

            # Read the clustering experiment log file
            log_exp = interpret_dict(exp)

            # Retrieve the experiment name from the log
            exp_name = log_exp["config_clustering"]["exp_name"]

            # Retrieve the clusterings from the log
            clusterings = log_exp["log_clustering"]["clusterings"]

            # Retrieve scaler
            scaler_class = log_exp["config_clustering"][exp_name]["scaler"]
            scaler_kw = log_exp["config_clustering"][exp_name]["scaler_kw"]
            if scaler_class is None:
                scaler = None
            else:
                scaler = scaler_class(**scaler_kw)

            # ----------- Prepare CVI log --------------
            log_cvi_fname = f"{path_res}{exp_name}/{d}-CVI.json"

            log_cvi = {
                **log_exp,
                "log_CVI": {
                    "CVI_names": CVI_names,
                    "log_fname" : log_cvi_fname,
                }
            }

            # Check if we should re_use
            re_use = re_use_experiment(
                log_cvi, re_run=re_run, overwrite=overwrite
            )

            if re_use:
                # load already saved experiment
                log_cvi = interpret_dict(log_cvi_fname)

                # update main log
                for cvi in CVI_names:
                    log['log_CVI'][d][exp_name][cvi] = log_cvi["log_CVI"][cvi]["selected"]

            else:

                # Prepare main log to store the selected k for each exp and CVI
                log["log_CVI"][d][exp_name] = {}

                for cvi, cvi_config in exps_config["config_CVI"].items():


                    # ============ Compute CVI values =============
                    print(f" ================ {cvi} ================ ")
                    t_start_cvi = time.time()

                    cvi_instance = cvi_config["cvi"](**cvi_config["cvi_init_kw"])

                    cvi_values = compute_all_scores(
                        cvi_instance,
                        data,
                        clusterings,
                        ts_dist=ts_dist,
                        scaler=scaler,
                        rng=config["config_CVI"]["seed"],
                        cvi_kwargs=cvi_config["cvi_kw"],
                    )

                    # if all cvi values were None, no k selected
                    try:
                        k_selected = cvi_instance.select(cvi_values)
                    except SelectionError as e:
                        k_selected = None

                    # ----------- Update logs --------------
                    # Print cvi information
                    for k, cvi_value in cvi_values.items():
                        print(k, cvi_value, flush=True)
                    print(f"Selected k: {k_selected} | True k: {k_ref}", flush=True)

                    t_end_cvi = time.time()
                    dt = t_end_cvi - t_start_cvi
                    print('Code executed in %.2f s' %(dt))

                    log_cvi[cvi] = {
                        "cvi_values" : cvi_values,
                        "selected" : k_selected,
                        "time" : dt,
                    }

                    # Update main log with the selected k for this CVI and dataset
                    log['log_CVI'][d][exp_name][cvi] = k_selected



                # ----------- Save experiment log ------------------
                save_log(
                    f"{log_cvi_fname}", log_cvi,
                    overwrite=True, add_date=False, new_name=True, verbose=False,
                )

            # ------------- Update main log ---------------------
            # Add current CVI log filename to the main log file
            log["log_CVI"]["path_CVI_files"].append(log_cvi_fname)

    # -------------------- Finalize log and save -----------------------
    t_end = time.time()
    dt = float(f"{t_end - t_start:.2f}")
    print(f"\n\nTotal execution time: {dt:.2f}s")

    log['log_CVI'].update(filtered_exp)
    log['log_CVI'].update(filtered_datasets)
    log['log_CVI']["time"] = dt
    save_log(
        f"{log_fname}.json", log,
        overwrite=True, add_date=False, new_name=True, verbose=False,
    )
    # Make the log visible in the output file
    print_log(log)

    fout.close()
    return log
