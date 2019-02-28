"""Common type hints.

Contains various commonly used types. Can be used in combination with
typing.TYPE_CHECKING to avoid loading snips and spacy early.
"""
from typing import TYPE_CHECKING, Any, Dict

from snips_nlu import SnipsNLUEngine
from spacy.language import Language
from spacy.tokens.doc import Doc
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from r4a_nao_nlp.engines import SnipsResult
from r4a_nao_nlp.graph import Graph

assert TYPE_CHECKING

# TODO: document what we use this type for: Can be converted to human-readable string and
# to an eobject.
# TODO: unique type for SNIPS parse result to distinguish
JsonDict = Dict[str, Any]

# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
