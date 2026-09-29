"""Additional util fonctions not related to datasets, config or logs"""

import numpy as np
import json
from pathlib import Path
import inspect
import importlib

from typing import List, Dict, Tuple, Union, Sequence, Any




def get_obj_from_string(obj_str: str) -> Any:
    """
    Return the object corresponding to a string representing it.

    The object can be a class or a function. The string must be of the
    form `"package.module.class"` or `"package.module.function"`.

    Both the hidden and non-hidden module names are supported.
    The hidden module name is the one that is used when importing a
    class or function

    The corresponding class or function will be imported.

    If the string is not of the correct form, an ImportError will be
    raised.

    Parameters
    ----------
    obj_str : str
        String representing the object to be imported.

    Returns
    -------
    Any
        The object corresponding to the string.
    """
    # rsplit(".", 1) will split once, at the very last occurence
    module_path, obj_name = obj_str.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, obj_name)

def class_to_string(cls):
    """
    Return the fully qualified import path for a class.

    Note that this can yield the "hidden" class of an object, with
    hidden module names

    Parameters
    ----------
    cls : type
        Class object to convert to a module-qualified string.

    Returns
    -------
    str
        Import path in the form ``"package.module.ClassName"``.
    """
    return f"{cls.__module__}.{cls.__name__}"

def obj_to_string(obj):
    """
    Return a readable string describing an object's concrete class.

    Note that this can yield the "hidden" class of an object, with
    hidden module names

    Parameters
    ----------
    obj : Any
        Object instance to describe.

    Returns
    -------
    str
        String in the form ``"Instance of package.module.ClassName"``.
    """
    cls = obj.__class__
    return f"Instance of {cls.__module__}.{cls.__name__}"

def serialize(obj):
    """
    Serialize an object to a JSON-compatible format.

    Parameters
    ----------
    obj : Any
        Object to serialize.

    Returns
    -------
    Any
        JSON-serializable representation of ``obj``. Sequences and
        dictionaries are converted recursively, NumPy arrays are turned
        into lists, and classes or functions are written as importable
        strings.
    """
    # To check if the object is serializable
    try:
        json.dumps(obj)
        return obj
    except (TypeError, OverflowError):
        if isinstance(obj, Sequence):
            res = [serialize(x) for x in obj]
        # If we are dealing with a dict of object
        elif isinstance(obj, dict):
            res = {key : serialize(item) for key, item in obj.items()}
        # If we are dealing with a dict of object
        elif isinstance(obj, np.ndarray):
            res = obj.tolist()
        else:
            # True for classes/types, False for instances
            if inspect.isclass(obj):
                res = class_to_string(obj)
            elif inspect.isfunction(obj):
                res = f"{obj.__module__}.{obj.__name__}"
            else:
                res = obj_to_string(obj)
        return res


def simplify_dict(config:dict) -> dict:
    """
    Translate a dictionary to a JSON-serializable dictionary.

    Parameters
    ----------
    config : dict
        Dictionary whose values may include arrays, classes, functions,
        or nested structures.

    Returns
    -------
    dict
        Copy of ``config`` where values unsupported by JSON are replaced
        by serialized representations.
    """
    #normal_types = [Sequence, str, list, dict, int, float, bool, type(None)]

    simpler_dict = {}
    for k, v in config.items():
        # Serialize value if necessary
        simpler_dict[k] = serialize(v)
    return simpler_dict

def interpret_dict(config:Union[dict, str]) -> dict:
    """
    Translate a given dict to a config dict, using classes and functions

    Make sure that classes and functions are written as
    ``package.module.class``.

    This function works for dict that are not config (and will not
    complement with default values). For a function specially designed
    for config dict, see `interpret_config` (which will complement
    with default values).

    This function will convert to int any key that is a string
    containing only digits.

    Parameters
    ----------
    config : Union[dict, str]
        Dictionary to interpret, or path to a JSON file containing one.

    Returns
    -------
    dict
        Dictionary where importable strings have been replaced with the
        corresponding Python objects.
    """
    if isinstance(config, str):
        if not config.endswith(".json"):
            config += ".json"
        config = load_json(config)

    interpreted_dict = {}
    for k, v in config.items():

        # If the key is a digit, read it as an integer, else keep it as a string
        if isinstance(k, str) and k.isdigit():
            new_k = int(k)
        else:
            new_k = k

        # If the value is a string, try to interpret it as a class or function
        if isinstance(v, str):
            try:
                interpreted_dict[new_k] = get_obj_from_string(v)
            # If there was an error, it's probably because this was a regular
            # string, not a string representing a class or function
            except (ImportError, AttributeError, ValueError):
                interpreted_dict[new_k] = v
        # If the value is a dict, recursively interpret it
        elif isinstance(v, dict):
            interpreted_dict[new_k] = interpret_dict(v)
        # Else, assume that the format is correct and keep the value as is
        else:
            interpreted_dict[new_k] = v

    return interpreted_dict

def equal_dict(d1: dict, d2:dict) -> bool:
    """
    Check whether 2 dicts are equal, allowing for lists in values
    """
    are_equal = True


def load_json(fname: str) -> Dict:
    """
    Load a JSON file and cast numeric string keys to integers.

    Parameters
    ----------
    fname : str
        Path to the JSON file.

    Returns
    -------
    Dict
        Parsed JSON dictionary with digit-only keys converted to
        integers.
    """
    def object_hook(json_dict):
        return {
            int(k) if k.isdigit() else k: v
            for (k, v) in json_dict.items()
        }
    with open(fname) as f_json:
        d = json.load(f_json, object_hook=object_hook)
    return d

def write_json(fname: str, data: Dict) -> None:
    """
    Write a dictionary to a JSON file.

    Parameters
    ----------
    fname : str
        Path to the JSON file.
    data : Dict
        Dictionary to write to the JSON file.
    """
    p = Path(fname)
    p.parent.mkdir(parents=True, exist_ok=True)
    json_str = json.dumps(data, indent=2)
    with open(fname, 'w', encoding='utf-8') as f:
        f.write(json_str)




