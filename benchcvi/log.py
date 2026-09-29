"""
Module for logs and log checks
"""

import numpy as np
from numpy.typing import NDArray
import os
import json
from pathlib import Path
from datetime import datetime
from copy import deepcopy

from typing import List, Dict, Tuple, Union, Any

from .utils import interpret_dict, simplify_dict
from .exceptions import InconsistentLogError
from .data import all_summary_stats


def save_log(
        log_fname: str,
        log_dict: dict[str, Any],
        overwrite: bool = False,
        add_date: bool = True,
        new_name: bool = False,
        verbose: bool = False,
    ) -> str | None:
    """Save a JSON log file and optionally rename it when a file exists.

    Parameters
    ----------
    log_fname : str
        Target file path for the log file, typically a ``.json`` filename.
    log_dict : dict[str, Any]
        Dictionary of log metadata to serialize. Typical keys include run
        parameters, status flags, and summary values. The ``log_filename`` key
        is added or updated before writing.
    overwrite : bool, default=False
        Whether to overwrite an existing log file at ``log_fname``.
    add_date : bool, default=True
        Whether to append a ``YYYY-MM-DD--HH:MM:SS`` timestamp to the filename
        before saving.
    new_name : bool, default=False
        Whether to create a new timestamped filename when ``log_fname`` already
        exists and ``overwrite`` is ``False``.
    verbose : bool, default=False
        Whether to print status messages during save operations.

    Returns
    -------
    str | None
        The final log filename that was written, or ``None`` if saving was
        skipped.
    """

    if add_date:
        # Create a new filename by adding a suffix to the original filename
        ext = Path(log_fname).suffix
        base = log_fname.split(ext)[0]
        full_date = datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
        log_fname = f"{base}-{full_date}{ext}"

    if os.path.isfile(log_fname):

        # Common message if log exists
        if int(verbose) > 0:
            print(f"Log file {log_fname} already exists.")
        # Added message if log should be overwritten
        if overwrite:
            if int(verbose) > 0:
                print(f"Overwriting log file {log_fname}.")
        # Added message if we then cancel the saving of the log file
        elif not new_name:
            log_fname = None
            if int(verbose) > 0:
                print(f"Not overwriting nor creating new filename. Skipping log saving.")

        # If we create a new name
        else:

            # Create a new filename by adding a suffix to the original filename
            ext = Path(log_fname).suffix
            base = log_fname.split(ext)[0]
            full_date = datetime.today().strftime('%Y-%m-%d--%H:%M:%S')
            log_fname = f"{base}-{full_date}{ext}"

            if int(verbose) > 0:
                print(f"Creating new filename: {log_fname}.")

    if log_fname is not None:

        # Make sure path exists otherwise create it
        p = Path(log_fname)
        p.parent.mkdir(parents=True, exist_ok=True)

        log_dict["log_filename"] = log_fname

        log_dict_serializable = simplify_dict(log_dict)

        with open(log_fname, 'w') as f_log:
            json.dump(log_dict_serializable, f_log, indent=2)

        if int(verbose) > 0:
            print(f"Saved log file to {log_fname}.")

    return log_fname


def print_log(
        log:dict,
    ) -> None:
    """
    Print a log dictionary in a readable text block.

    Adds a `START LOG` and `END LOG` markers to the output to easily extract the log from a text file (see :func:`extract_log_from_text`).

    Parameters
    ----------
    log : dict
        Log dictionary to serialize and print.

    Returns
    -------
    None
        This function prints to standard output and returns nothing.
    """
    simpler_dict = simplify_dict(log)
    print(f"\n┌─{'─'*70}─┐")
    if log['log_filename'] is None:
        print(f"{" "*3} Log file: NOT SAVED YET")
    else:
        print(f"{" "*3} Log file: {log['log_filename']}")
    print(f"START LOG")
    print(json.dumps(simpler_dict, indent=2), flush=True)
    print(f"END LOG")
    print(f"\n└─{'─'*70}─┘", flush=True)

def extract_log_from_text(fname) -> list[dict]:
    """
    Extract log dicts from a log text file.

    The text file should contain a JSON string representing the log dict,
    starting with a line containing "START LOG" and ending with a line
    containing "END LOG".

    There could be several logs in the same text file, each one starting with "START LOG" and ending with "END LOG", but logically, there should be only 2 logs per text file, the one at the very beginning and the one at the very end, with some other text in between.

    Parameters
    ----------
    fname : str
        Path to the log text file.

    Returns
    -------
    list[dict]
        Log dictionaries extracted from the text file, in the order in
        which they appear.
    """
    with open(fname, 'r') as f:
        lines = f.readlines()
    l_i_start = [i for i, line in enumerate(lines) if line == "START LOG\n"]
    l_i_end = [i for i, line in enumerate(lines) if line == "END LOG\n"]
    l_logs = []
    for i_start, i_end in zip(l_i_start, l_i_end):
        json_lines = [l for l in lines[i_start + 1:i_end]]
        json_str = "".join(json_lines)
        json_dict = json.loads(json_str)
        l_logs.append(interpret_dict(json_dict))
    return l_logs


def extract_keys_from_log(log: Union[dict, str]) -> dict:
    """
    Extract the keys from a log dictionary or a log JSON file.

    Parameters
    ----------
    log : Union[dict, str]
        Log dictionary or path to a log JSON file.

    Returns
    -------
    dict
        Dictionary containing the keys of the log dictionary, with nested
        dictionaries for nested keys.
    """
    log = interpret_dict(log)

    keys = {}

    for k, v in log.items():
        if isinstance(v, dict):
            keys[k] = extract_keys_from_log(v)
        else:
            keys[k] = k
    return keys

def consistent_data_summary_stats(
        stats1:dict,
        stats2:dict,
        log_filename: str = None,
        dataset_name :str = None,
    ):
    """
    Check that 2 (current, log) data summary statistics are consistent

    Some values should be equal and some should be close enough.

    Parameters
    ----------
    stats1 : dict
        Current data summary statistics.
    stats2 : dict
        Log data summary statistics.
    log_filename : str, optional
        Name of the log file.
    dataset_name : str, optional
        Name of the dataset.

    Raises
    ------
    InconsistentLogError
        If the summary statistics are inconsistent.
    """

    msg = f"Summary stats are inconsistent."

    if log_filename is not None:
        msg += f" In log {log_filename}."
    if dataset_name is not None:
        msg += (
            f" Problem with dataset {dataset_name}."
        )

    # ----------- Keys that should have equal values ----------
    equal_keys = ["shape", "n_labels", "datapoints_per_label", "sum_idx_per_label"]

    for key in equal_keys:
        if stats1[key] != stats2[key]:

            msg += (
                f" For key {key}, found "
                + f"{stats1[key]} but log has {stats2[key]}"
            )

            raise InconsistentLogError(msg)

    # ---- Keys that should have np.isclose values -----------
    close_keys = ["mean", "std", "min", "max"]

    for key in close_keys:
        if not np.allclose(stats1[key], stats2[key]):
            msg += (
                f" For key {key}, found "
                + f"{stats1[key]} but log has {stats2[key]}"
            )

            raise InconsistentLogError(msg)

def consistent_log_data(
    log: dict,
    summary_stats: dict = None,
) -> dict:
    """
    Check if a log data is consistent with the current state of the data files.

    Parameters
    ----------
    log : dict
        Log dictionary containing information about the datasets and their summary statistics.
    summary_stats : dict, optional
        Precomputed summary statistics for the datasets. If not provided, they will be computed from the current data files.

    Return
    ------
    dict
        The summary statistics for the datasets, either provided or computed.

    Raises
    ------
    InconsistentLogError
        If any inconsistency is found between the log and the current state of the data files.
    """

    path_data = log["config_data"]["path_data"]

    # -------------- Check that all dataset files exist ----------------
    for d in log["log_data"]["kept_datasets"]:
        if not os.path.isfile(f"{path_data}{d}"):
            raise InconsistentLogError(f"Dataset file {path_data}{d} does not exist but is referenced in the log {log["log_filename"]}")

    # ------- Check that summary stats in log_data are consistent -------

    # Compute current summary stats if not provided
    if summary_stats is None:
        summary_stats = all_summary_stats(
            path_data, log["log_data"]["kept_datasets"]
        )

    for d, summary in summary_stats.items():
        consistent_data_summary_stats(
            summary, log["log_data"][d],
            log_filename=log["log_filename"],
            dataset_name=d
        )

    return summary_stats

def consistent_log_clustering(
    log: dict,
) -> None:
    """
    Check if a log clustering is consistent with the current state of
    the clustering experiment files.

    This function should be called after having called consistent_log_data

    Parameters
    ----------
    log : dict
        Log dictionary containing information about the datasets and their summary statistics.

    Raises an InconsistentLogError if any inconsistency is found.
    """

    # -------------- Check that all experiment files exist ----------------
    for f in log["log_clustering"]["path_exp"]:
        if not os.path.isfile(f):
            raise InconsistentLogError(f"Experiment file {f} does not exist but is referenced in the log {log["log_filename"]}")

        log_exp = interpret_dict(f)

        # Check that summary stats in log exp consistent with log_data
        # (and presumably log_data has already been check against the current
        # file status by now when calling consistent log_data)
        dataset = log_exp["info_dataset"]["dataset_name"]
        consistent_data_summary_stats(
            log_exp["info_dataset"][dataset], log["log_data"][dataset], log_filename=log["log_filename"], dataset_name=dataset)




def consistent_log_exp(
    log1:dict, log2: dict
) -> bool:
    """
    Compare a saved log file (log2) with the current experiment (log1).

    Works for clustering exp and CVI exp.

    For clustering exp, checks dataset consistency and config_clustering
    consistency (and then log_clustering is what we re-use).

    For CVI exp, in addition config_CVI consistency and log_clustering
    consistency (and then we re-use both log_clustering and log_CVI).

    In any case, log_filename, is ignored in the comparison.

    Parameters
    ----------
    log1 : dict
        Current experiment log dictionary.
    log2 : dict
        Saved log dictionary to compare against.

    Returns
    -------
    bool
        True if the logs are consistent, False otherwise.
    """

    is_equal = True

    d = log1["info_dataset"]["dataset_name"]

    # Deep copy because we will do some popping
    new_log = deepcopy(log1)
    saved_log = deepcopy(log2)

    # ------------------- Dataset consistency -----------------------
    # First pop data summary stats to treat them separately
    # because of the difference between "is_close" and exact comparison
    stats1 = new_log["info_dataset"].pop(d)
    stats2 = saved_log["info_dataset"].pop(d)

    # Check that summary stats are equal between the 2 log exp
    try:
        consistent_data_summary_stats(
            stats1, stats2,
            log_filename=saved_log["log_filename"],
            dataset_name=d
        )
    except InconsistentLogError as e:
        is_equal = False

    if new_log["info_dataset"] != saved_log["info_dataset"]:
        is_equal = False

    # ------------------- Config Clustering consistency ----------------
    if "config_clustering" in new_log:
        if new_log["config_clustering"] != saved_log["config_clustering"]:
            is_equal = False

    # -------------------- Config CVI consistency ----------------------
    if "config_CVI" in new_log:

        # Check that config_CVI is equal
        if new_log["config_CVI"] != saved_log["config_CVI"]:
            is_equal = False
        if new_log["log_clustering"] != saved_log["log_clustering"]:
            is_equal = False


    return is_equal

