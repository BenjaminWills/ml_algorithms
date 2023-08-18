import logging
import os

import coloredlogs

coloredlogs.install()


def add_file_output(logger: logging.Logger, logging_path: str, formatter_style: str):
    """Helper function to add a file output to a logger

    Parameters
    ----------
    logger : logging.Logger
        A logger object
    logging_path : str
        A path for the logfiles to be saved to
    formatter_style : str
        The style of the logs that are outputted. For inspiration check out: @https://docs.python.org/3/library/logging.html#formatter-objects
    """
    file_handler = logging.FileHandler(logging_path)
    formatter = logging.Formatter(formatter_style)
    stream_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


def make_logger(
    logging_path: str = os.getcwd() + "/" + __name__ + ".log",
    save_logs: bool = True,
    formatter_style: str = "%(asctime)s - %(name)s - %(levelname)s :: %(message)s",
    logger_name: str = __name__,
    logger_level: int = logging.INFO,
) -> logging.Logger:
    """A function to make a custom logger

    Parameters
    ----------
    logging_path : str, optional
        A path for the logfiles to be saved to,
        by default os.getcwd()+"/"+__name__+".log"
    save_logs : bool, optional
        If True, will save logs to the logging path with the given logging format,
        by default True
    formatter_style : _type_, optional
        The style of the logs that are outputted. For inspiration check out: @https://docs.python.org/3/library/logging.html#formatter-objects,
        by default "%(asctime)s - %(name)s - %(levelname)s :: %(message)s"
    logger_name : str, optional
        A name for the logger,
        by default __name__
    logger_level : int, optional
        The minimum severity level of the logger,
        by default logging.INFO

    Returns
    -------
    logging.Logger
        A logger with the specified qualities
    """

    # Create logger
    logger = logging.getLogger(name=logger_name)
    # Set urgency level
    logger.setLevel(logger_level)

    if save_logs:
        add_file_output(logger, logging_path, formatter_style)

    return logger
