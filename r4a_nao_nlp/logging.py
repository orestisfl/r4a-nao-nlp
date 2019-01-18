# TODO: docstrings
# vim:ts=4:sw=4:expandtab:fo-=t
import logging

_handler = logging.StreamHandler()


def init():
    logging.basicConfig()
    original_handler = logging.root.handlers[0]
    original_handler.setLevel(logging.WARNING)
    original_handler.set_name("original")

    _handler.set_name("r4a_nao_nlp")
    _handler.formatter = original_handler.formatter


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.addHandler(_handler)
    return logger


def set_level(level: int) -> None:
    _handler.setLevel(level)
    logging.root.setLevel(level)
