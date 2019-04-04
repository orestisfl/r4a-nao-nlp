"""Miscellaneous utilities."""
import argparse
import logging
import sys
from functools import wraps
from itertools import chain, combinations
from typing import Callable, Collection, Iterable, Iterator, Optional

_LOGGING_HANDLER = logging.StreamHandler()


def init_logging() -> None:
    """Create a logging handler for our modules and configure the root logger."""
    logging.basicConfig(format="%(asctime)s " + logging.BASIC_FORMAT)
    original_handler = logging.root.handlers[0]
    original_handler.setLevel(logging.WARNING)
    original_handler.set_name("original")

    _LOGGING_HANDLER.set_name("r4a_nao_nlp")
    _LOGGING_HANDLER.formatter = original_handler.formatter


def create_logger(name: str) -> logging.Logger:
    """Create logger, adding the common handler."""
    if name is None:
        raise TypeError("name is None")

    logger = logging.getLogger(name)
    # Should be unique
    logger.addHandler(_LOGGING_HANDLER)
    return logger


class ArgumentParser(argparse.ArgumentParser):
    """Helper object that wraps argparse.ArgumentParser to provide common operations
    used in this package."""

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
        _LOGGING_HANDLER.setLevel(level)
        logging.root.setLevel(level)

        if self._ecore_dest:
            from r4a_nao_nlp.ecore import init_root

            init_root(getattr(namespace, self._ecore_dest))

        return namespace

    def add_ecore_root_argument(self, *args, **kwargs):
        kwargs.setdefault("default", "highLevelNaoApp.ecore")
        kwargs.setdefault("help", "Path to the root Ecore meta-model")

        self._ecore_dest = self.add_argument(*args, **kwargs).dest


def timed(fun: Callable, logger: Optional[logging.Logger] = None) -> Callable:
    """Execution time to log decorator.

    If `logger` is `None`, the `"logger"` attribute of the function's module is used.
    """
    from inspect import getmodule
    from time import time

    module = getmodule(fun)
    logger = logger or module.logger

    @wraps(fun)
    def wrapper(*args, **kwargs):
        start = time()
        result = fun(*args, **kwargs)
        diff = time() - start
        logger.debug(
            "Function %s.%s took %.2f seconds to complete",
            module.__name__,
            fun.__qualname__,
            diff,
        )
        return result

    return wrapper


def other_isinstance(fun: Callable, instance: Optional[type] = None):
    """Argument instance checker decorator."""

    @wraps(fun)
    def wrapper(arg1, arg2, *args, **kwargs):
        if instance is None:
            if not isinstance(arg2, type(arg1)):
                return NotImplemented
        elif not isinstance(arg2, instance):
            return NotImplemented

        return fun(arg1, arg2, *args, **kwargs)

    return wrapper


class PowerSet:
    """A reversible iterable that produces all possible r-length combinations of a
    collection's elements inside a specified range of r values.

    - `r_stop` is the maximum r value to use. If `r_stop <= 0`, its absolute value will
      be subtraced from the `collection`'s length.
    - `r_start` is the minimum r value to use. If `r_start < 0`, its absolute value will
      be subtraced from the `collection`'s length.
    """

    def __init__(self, collection: Collection, r_stop: int = 0, r_start: int = 0):
        self._stop = r_stop if r_stop > 0 else len(collection) + 1 + r_stop
        self._start = r_start if r_start >= 0 else len(collection) + 1 + r_start
        self._collection = collection

    def __iter__(self):
        return self._iter(range(self._start, self._stop))

    def __reversed__(self):
        return self._iter(reversed(range(self._start, self._stop)))

    def _iter(self, r_range: Iterable) -> Iterator:
        return iter(
            chain.from_iterable(combinations(self._collection, r) for r in r_range)
        )


def before_37() -> bool:
    """Check if python version is before 3.7"""
    major, minor, _, _, _ = sys.version_info

    return (major, minor) < (3, 7)


# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
