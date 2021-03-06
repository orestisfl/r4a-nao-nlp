from string import ascii_lowercase
from typing import List, Optional
from unittest.mock import MagicMock, Mock

import networkx as nx
import pytest
from spacy.tokens.span import Span
from spacy.tokens.token import Token

from r4a_nao_nlp import subsentence
from r4a_nao_nlp.engines import shared
from r4a_nao_nlp.graph import Graph


@pytest.fixture(scope="module", autouse=True)
def init():
    code = shared.init.__wrapped__.__code__
    kwargs = {a: None for a in code.co_varnames[: code.co_argcount] if a != "self"}
    # We only need spacy
    kwargs["spacy_lang"] = "en_core_web_sm"
    print(f"shared.init with kwargs = {kwargs}")
    shared.init(**kwargs)

    mock = MagicMock(score=1.0)
    mock.__float__.return_value = 1.0
    mock.__gt__.return_value = True
    mock.__le__.return_value = False
    mock.__lt__.return_value = False
    shared.parse = Mock(return_value=mock)


def create_spacy_sent(length: int, s: Optional[str] = None) -> Span:
    """Return a fake spacy sentence of the given length."""
    if s is None:
        s = " ".join(idx_to_word(idx) for idx in range(length - 1)) + " ."

    doc = shared.spacy(s)
    sent = next(doc.sents)
    assert len(sent) == length
    return sent


def idx_to_word(idx: int) -> str:
    """Return a word to be used for the given position of the sentence."""
    return ascii_lowercase[idx % 26] * (idx // 26 + 1)


def test_common_modifier():
    all_tags = [
        ["B-V", "O", "O", "O", "B-ARGM-TMP", "I-ARGM-TMP"],
        ["O", "O", "B-V", "B-ARG1", "B-ARGM-TMP", "I-ARGM-TMP"],
        ["O", "O", "O", "O", "O", "B-V"],
    ]
    sent = create_spacy_sent(6)
    s = subsentence.create_subsentences(all_tags, sent)

    assert s[0].compatible == {s[1], s[2]}
    assert s[1].compatible == {s[0], s[2]}
    assert s[0].modifiers == {s[2]: (s[2].verb, s[0].argms["ARGM-TMP"])}
    assert s[1].modifiers == {s[2]: (s[2].verb, s[1].argms["ARGM-TMP"])}
    assert s[2].modifiers == {}
    assert s[2].modifying == {
        s[0]: (s[2].verb, s[0].argms["ARGM-TMP"]),
        s[1]: (s[2].verb, s[1].argms["ARGM-TMP"]),
    }


def test_one_modifier():
    all_tags = [
        ["B-V", "B-ARGM-TMP", "I-ARGM-TMP", "O", "O", "O"],
        ["O", "O", "B-V", "O", "O", "O"],
        ["O", "O", "O", "O", "B-V", "B-ARG1"],
    ]
    sent = create_spacy_sent(6)
    s = subsentence.create_subsentences(all_tags, sent)

    assert s[0].compatible == {s[1], s[2]}
    assert s[1].compatible == {s[0], s[2]}
    assert s[0].modifiers == {s[1]: (s[1].verb, s[0].argms["ARGM-TMP"])}
    assert s[1].modifiers == {}
    assert s[1].modifying == {s[0]: (s[1].verb, s[0].argms["ARGM-TMP"])}
    assert s[2].modifiers == {}


def test_incompatible():
    all_tags = [
        ["B-V", "O", "O", "B-ARG1", "I-ARG1", "I-ARG1"],
        ["B-V", "O", "O", "B-ARG1", "I-ARG1", "I-ARG1"],
        ["O", "O", "B-V", "B-ARG1", "I-ARG1", "I-ARG1"],
        ["O", "O", "O", "O", "B-V", "O"],
    ]
    sent = create_spacy_sent(6)
    s = subsentence.create_subsentences(all_tags, sent)

    assert s[0].compatible == {s[2]}
    assert s[2].compatible == {s[0], s[1]}
    assert s[3].compatible == set()

    assert all(sub.argms == {} for sub in s)
    assert all(sub.modifiers == {} for sub in s)


def test_single_combination():
    all_tags = [
        ["B-V", "O", "O", "O", "B-ARGM-TMP", "I-ARGM-TMP"],
        ["O", "O", "B-V", "B-ARG1", "B-ARGM-TMP", "I-ARGM-TMP"],
        ["O", "O", "O", "O", "O", "B-V"],
    ]
    sent = create_spacy_sent(6)
    s = subsentence.create_subsentences(all_tags, sent)
    c = list(subsentence.create_combinations_from_subsentences(s))

    assert len(c) == 1
    assert len(c[0]) == 3
    assert all(tags in [sub.tags for sub in c[0]] for tags in all_tags)


def test_two_combinations():
    all_tags = [
        ["B-V", "O", "O", "B-ARG1", "I-ARG1", "I-ARG1"],
        ["B-V", "O", "O", "B-ARG1", "I-ARG1", "O"],
        ["O", "O", "B-V", "B-ARG1", "I-ARG1", "I-ARG1"],
        ["O", "B-V", "O", "O", "O", "O"],
    ]
    sent = create_spacy_sent(6)
    s = subsentence.create_subsentences(all_tags, sent)
    c = list(subsentence.create_combinations_from_subsentences(s))

    assert len(c) == 2
    tags1 = [sub.tags for sub in c[0]]
    tags2 = [sub.tags for sub in c[1]]

    assert (all_tags[0] in tags1) ^ (all_tags[0] in tags2)
    assert (all_tags[1] in tags1) ^ (all_tags[1] in tags2)
    assert all_tags[2] in tags1
    assert all_tags[2] in tags2
    assert all_tags[3] in tags1
    assert all_tags[3] in tags2


def test_multiple_combinations():
    all_tags = [
        ["B-V", "O", "O", "O", "O"],
        ["O", "B-V", "B-ARG1", "I-ARG1", "I-ARG1"],
        ["O", "O", "B-V", "B-ARG1", "O"],
        ["O", "B-V", "O", "O", "O"],
        ["O", "B-ARG1", "I-ARG1", "I-ARG1", "B-V"],
        ["O", "O", "O", "O", "B-V"],
    ]
    sent = create_spacy_sent(5)
    s = subsentence.create_subsentences(all_tags, sent)
    c = list(subsentence.create_combinations_from_subsentences(s))
    c_tags = [[sub.tags for sub in comb] for comb in c]

    assert len(c) == 3
    assert len(c[0]) == 4
    assert all_tags[0] in c_tags[0]
    assert all_tags[2] in c_tags[0]
    assert all_tags[3] in c_tags[0]
    assert all_tags[5] in c_tags[0]
    assert len(c[1]) == 2
    assert all_tags[0] in c_tags[1]
    assert all_tags[1] in c_tags[1]
    assert len(c[1]) == 2
    assert all_tags[0] in c_tags[2]
    assert all_tags[4] in c_tags[2]


def verify_graph(
    g: Graph, s: List[subsentence.SubSentence], all_tags: List[List[str]]
) -> None:
    for node, data in g.nodes(data=True):
        if "idx" in data:
            idx = data["idx"]
            assert node.tags == all_tags[idx]
            assert node == s[idx]
        else:
            assert isinstance(node, str)
            assert node in ("Start", "End-0")

    # Assume only single sentence:
    assert len(g) == len(s) + 2
    assert g.sent_idx == 0
    assert g.sent_end == "End-0"
    assert g.sent_end in g
    assert len(g[g.sent_end]) == 0
    predecessors = list(g.predecessors(g.sent_end))
    assert len(predecessors) == 1
    assert g.nodes[predecessors[0]]["idx_main"] == max(
        nx.get_node_attributes(g, "idx_main").values()
    )


def cmp_tokens_to_indices(tokens: List[Token], *indices: int) -> bool:
    return cmp_tokens_to_words(tokens, *(idx_to_word(idx) for idx in indices))


def cmp_tokens_to_words(tokens: List[Token], *words: str) -> bool:
    assert all(isinstance(t, Token) for t in tokens)
    return [t.text for t in tokens] == list(words)


def test_graph():
    all_tags = [
        ["B-V", "B-ARGM-TMP", "I-ARGM-TMP", "O", "O", "O"],
        ["O", "O", "B-V", "B-ARG1", "O", "O"],
        ["O", "O", "O", "B-ARG1", "B-V", "O"],
    ]
    sent = create_spacy_sent(6)
    s = subsentence.create_subsentences(all_tags, sent)
    c = list(subsentence.create_combinations_from_subsentences(s))
    assert len(s) == 3
    assert len(c) == 1

    g = c[0].to_graph()
    verify_graph(g, s, all_tags)

    adj = g[s[0]]
    assert len(adj) == 1
    assert not adj[s[2]]["label"]

    adj = g[s[1]]
    assert len(adj) == 1
    assert cmp_tokens_to_indices(adj[s[0]]["label"], 1)

    adj = g[s[2]]
    assert len(adj) == 1
    assert cmp_tokens_to_words(adj["End-0"]["label"], ".")


def test_graph_common_argms():
    all_tags = [
        [
            "B-V",
            "O",
            "O",
            "B-ARG1",
            "I-ARG1",
            "I-ARG1",
            "B-ARGM-TMP",
            "I-ARGM-TMP",
            "O",
        ],
        [
            "O",
            "O",
            "B-V",
            "B-ARG1",
            "I-ARG1",
            "I-ARG1",
            "B-ARGM-TMP",
            "I-ARGM-TMP",
            "O",
        ],
        ["O", "O", "O", "O", "O", "O", "O", "B-V", "O"],
    ]
    sent = create_spacy_sent(9, "Wave and extend your left arm while running.")
    s = subsentence.create_subsentences(all_tags, sent)
    c = list(subsentence.create_combinations_from_subsentences(s))
    assert len(s) == 3
    assert len(c) == 1

    g = c[0].to_graph()
    verify_graph(g, s, all_tags)

    adj = g[s[0]]
    assert len(adj) == 1
    assert cmp_tokens_to_words(adj[s[1]]["label"], "and")

    adj = g[s[2]]
    assert len(adj) == 2
    assert cmp_tokens_to_words(adj[s[0]]["label"], "while")
    assert cmp_tokens_to_words(adj[s[1]]["label"], "while")

    adj = g[s[1]]
    assert len(adj) == 1
    assert cmp_tokens_to_words(adj["End-0"]["label"], ".")


def test_graph_multiple_argms_negation():
    all_tags = [
        [
            "B-V",
            "O",
            "O",
            "B-ARG1",
            "I-ARG1",
            "B-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "O",
        ],
        [
            "O",
            "O",
            "B-V",
            "B-ARG1",
            "I-ARG1",
            "B-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "I-ARGM-TMP",
            "O",
        ],
        ["O", "O", "O", "O", "O", "O", "B-ARGM-NEG", "B-V", "B-ARG1", "O"],
    ]
    sent = create_spacy_sent(10, "Open and extend your arm while not saying hello.")
    s = subsentence.create_subsentences(all_tags, sent)
    c = list(subsentence.create_combinations_from_subsentences(s))
    assert len(s) == 3
    assert len(c) == 1

    g = c[0].to_graph()
    verify_graph(g, s, all_tags)

    adj = g[s[0]]
    assert len(adj) == 1
    assert cmp_tokens_to_words(adj[s[1]]["label"], "and")

    adj = g[s[2]]
    assert len(adj) == 2
    assert cmp_tokens_to_words(adj[s[0]]["label"], "while", "not")
    assert cmp_tokens_to_words(adj[s[1]]["label"], "while", "not")

    adj = g[s[1]]
    assert len(adj) == 1
    assert cmp_tokens_to_words(adj["End-0"]["label"], ".")


# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
