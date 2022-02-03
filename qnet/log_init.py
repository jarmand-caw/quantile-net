import logging
import os

logger = logging.getLogger(__name__)

def initialize_logger(to_file=False, log_name=None, output_dir=None):
    """

    Args:
        to_file: bool. If true, writes to lot file
        log_name: prefixes name of log file
        output_dir: where to write the file to. If none, will be in content root dir

    Returns:
        None
    """

    log_fmt = "[%(asctime)s%(levelname)8s], [%(filename)s:%(lineno)s "
    log_fmt += "- %(funcName)s()], %(message)s"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(log_fmt)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if to_file:
        log_name = log_name.replace(" ", "-").lower()

        if output_dir is not None:
            if not os.path.isdir(output_dir):
                raise FileNotFoundError("no directory {}".format(output_dir))
        else:
            output_dir = os.path.dirname(os.path.realpath(__file__))

        logger.info("log file: {}".format(log_name))

        handler = logging.FileHandler(os.path.join(output_dir, log_name), "w")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(log_fmt)
        handler.setFormatter(formatter)
        logger.addHandler(handler)