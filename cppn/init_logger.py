import logging
import colorlog


def init_logger(name):
    logger = logging.getLogger(name)
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter('%(log_color)s%(levelname)s:%(name)s:%(message)s'))
    logger = colorlog.getLogger('example')
    logger.addHandler(handler)
    logger.propagate = False
    return logger