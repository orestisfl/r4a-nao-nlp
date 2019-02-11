"""Common operations relating to logging and argument parsing."""
# vim:ts=4:sw=4:expandtab:fo-=t
import argparse
import logging

_logging_handler = logging.StreamHandler()


def init_logging() -> None:
    """Create a logging handler for our modules and configure the root logger."""
    logging.basicConfig()
    original_handler = logging.root.handlers[0]
    original_handler.setLevel(logging.WARNING)
    original_handler.set_name("original")

    _logging_handler.set_name("r4a_nao_nlp")
    _logging_handler.formatter = original_handler.formatter


def create_logger(name: str) -> logging.Logger:
    """Create logger, adding the common handler."""
    if name is None:
        raise TypeError("name is None")
    if name in logging.root.manager.loggerDict:
        raise ValueError("logger already exists")

    logger = logging.getLogger(name)
    logger.addHandler(_logging_handler)
    return logger


class ArgumentParser(argparse.ArgumentParser):
    """Helper object that wraps argparse.ArgumentParser to provide common operations used
    in this package."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument(
            "-v",
            "--verbose",
            action="count",
            default=0,
            help="increases log verbosity for each occurence.",
        )
        self._ecore_dest = None

    def parse_args(self, *args, **kwargs):
        namespace = super().parse_args(*args, **kwargs)

        level = max(3 - namespace.verbose, 0) * 10
        _logging_handler.setLevel(level)
        logging.root.setLevel(level)

        if self._ecore_dest:
            from r4a_nao_nlp.ecore import init_root

            init_root(getattr(namespace, self._ecore_dest))

        return namespace

    def add_ecore_root_argument(self, *args, **kwargs):
        kwargs.setdefault("default", "highLevelNaoApp.ecore")
        kwargs.setdefault("help", "Path to the root Ecore meta-model")

        self._ecore_dest = self.add_argument(*args, **kwargs).dest
