import logging
import matplotlib
import numpy as np
import pathlib
import time
from typing import Literal, Optional, Union

_valid_log_level_dict = {"debug": 10, "info": 20, "warning": 30, "error": 40, "critical": 50}

# from https://stackoverflow.com/a/35804945/1691778
def _add_log_level(levelname, levelvalue):
    levelname = levelname.lower()
    if hasattr(logging, levelname) or hasattr(logging.getLoggerClass(), levelname):
        print(f"{levelname} is already an existing log level. Skipping this step")
    else:
        def _log_for_level(self, msg, *args, **kwargs):
            if self.isEnabledFor(levelvalue):
                self._log(levelvalue, msg, args, **kwargs)
        
        def _log_to_root(msg, *args, **kwargs):
            logging.log(levelvalue, msg, *args, **kwargs)
        
        logging.addLevelName(levelvalue, levelname)
        setattr(logging, levelname, levelvalue)
        setattr(logging.getLoggerClass(), levelname, _log_for_level)
        setattr(logging, levelname, _log_to_root)

        _valid_log_level_dict[levelname] = levelvalue



def _validate_log_level(log_level: Union[str, int], log_location: str) -> int:

    if isinstance(log_level, str):
        if log_level in _valid_log_level_dict:
            log_level = _valid_log_level_dict[log_level]
        else:
            print(f"`{log_location}_log_level` must be one of {_valid_log_level_dict.keys()} or {_valid_log_level_dict.values()}, but got `{log_level}`")
            print(f"Setting `{log_location}_log_level` to `info` (20)")
            log_level = 20
    
    _key = min(_valid_log_level_dict, key=_valid_log_level_dict.get)
    _val = _valid_log_level_dict[_key]
    if log_location == "console" and log_level <= _val:
        print(f"Setting `{log_location}_log_level` to a level lower than {_key} or {_val} is not allowed")
        print(f"Setting `{log_location}_log_level` to `info` (20)")
        log_level = 20
    
    return log_level



def config_logger(console_log_level: Union[str, int],
                  file_log_level: Optional[Union[str, int]],
                  output_path: pathlib.Path,
                  output_filename_suffix: str):
    
    log = logging.getLogger()

    # If there are any existing handles (from previous runs), clear them
    if log.hasHandlers(): log.handlers.clear()

    log_message_format = logging.Formatter(
        "%(levelname)8s  >  %(funcName)s()  >  %(filename)s:%(lineno)-4s  >  %(message)s"
    )

    # Adding an additional log level 'trace' (to deal with
    # logging specific variable states during calculations
    # -- i.e. error tracing -- mostly from the Hamiltonian and
    # derivative calculations which would otherwise be
    # too lengthy and laggy to display in the main terminal)
    _add_log_level("trace", 5)

    # Create console handler (print logs to console)
    console_log_level = _validate_log_level(console_log_level, "console")
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)
    console_handler.setFormatter(log_message_format)
    log.addHandler(console_handler)

    # Create file handler (print logs to file)
    if file_log_level:
        file_log_level = _validate_log_level(file_log_level, "file")
        log_filename = output_path / f"scotty_log{output_filename_suffix}.log"
        file_handler = logging.FileHandler(log_filename, mode = "a", encoding = "utf-8")
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(log_message_format)
        log.addHandler(file_handler)
    
    # Setting the log level
    if file_log_level is None: log.setLevel(console_log_level)
    else: log.setLevel(min(console_log_level, file_log_level))
    
    # Suppress all matplotlib log messages (except those
    # which are important)
    matplotlib.set_loglevel("warning")



# RALT short for return_and_log_trace
def ralt(log: logging.Logger, msg: str, f: callable, *args, **kwargs):
    result = f(*args, **kwargs)
    if msg: log.trace(msg, result)
    return result



def timer(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    return result, duration



# MLN short for multi-line numbers
def mln(arr: Union[list, np.ndarray], prefix: str) -> str:
    if isinstance(arr, list): arr = np.array(arr)
    if arr.ndim != 1: raise ValueError(f"The array passed must be one-dimensional!")
    split_arr = np.array_split(arr, len(arr)//6 + 1)
    msg = f"\n{prefix}".join(f"{sub_arr}" for sub_arr in split_arr)
    return msg



# To make arrays have commas in them (for debug messages)
def arr2str(arr: Union[list, np.ndarray], precision: int = 7, separator: str = ", ") -> str:
    if isinstance(arr, list): arr = np.array(arr)
    msg = np.array2string(arr, precision=precision, separator=separator)
    return msg