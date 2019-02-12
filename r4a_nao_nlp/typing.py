"""Common type hints.

Contains various commonly used types. Can be used in combination with
typing.TYPE_CHECKING to avoid loading spacy early.
"""
from typing import Any, Dict

from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token

# TODO: document what we use this type for: Can be converted to human-readable string and
# to an eobject.
# TODO: unique type for SNIPS parse result to distinguish
JsonDict = Dict[str, Any]

# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
