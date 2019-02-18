from operator import itemgetter

import networkx as nx
import pytest

from r4a_nao_nlp.cli import process_document
from r4a_nao_nlp.engines import shared

SRL_CACHE = {
    "Open your left hand and then extend it while saying hello.": {
        "verbs": [
            {
                "verb": "Open",
                "description": "[V: Open] [ARG1: your left hand] and then extend it while saying hello .",
                "tags": [
                    "B-V",
                    "B-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "extend",
                "description": "Open your left hand and [ARGM-TMP: then] [V: extend] [ARG1: it] [ARGM-TMP: while saying hello] .",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-ARGM-TMP",
                    "B-V",
                    "B-ARG1",
                    "B-ARGM-TMP",
                    "I-ARGM-TMP",
                    "I-ARGM-TMP",
                    "O",
                ],
            },
            {
                "verb": "saying",
                "description": "Open your left hand and then extend it while [V: saying] [ARG1: hello] .",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-V",
                    "B-ARG1",
                    "O",
                ],
            },
        ],
        "words": [
            "Open",
            "your",
            "left",
            "hand",
            "and",
            "then",
            "extend",
            "it",
            "while",
            "saying",
            "hello",
            ".",
        ],
    },
    "Say hello and lay on your back": {
        "verbs": [
            {
                "verb": "Say",
                "description": "[V: Say] [ARG1: hello] and lay on your back",
                "tags": ["B-V", "B-ARG1", "O", "O", "O", "O", "O"],
            },
            {
                "verb": "lay",
                "description": "Say hello and [V: lay] [ARG2: on your back]",
                "tags": ["O", "O", "O", "B-V", "B-ARG2", "I-ARG2", "I-ARG2"],
            },
        ],
        "words": ["Say", "hello", "and", "lay", "on", "your", "back"],
    },
    "Turn on the leds of your chest and legs and go left": {
        "verbs": [
            {
                "verb": "Turn",
                "description": "[V: Turn] [ARGM-DIR: on] [ARG1: the leds of your chest and legs] and go left",
                "tags": [
                    "B-V",
                    "B-ARGM-DIR",
                    "B-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "I-ARG1",
                    "O",
                    "O",
                    "O",
                ],
            },
            {
                "verb": "go",
                "description": "Turn on the leds of your chest and legs and [V: go] [ARG1: left]",
                "tags": [
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "O",
                    "B-V",
                    "B-ARG1",
                ],
            },
        ],
        "words": [
            "Turn",
            "on",
            "the",
            "leds",
            "of",
            "your",
            "chest",
            "and",
            "legs",
            "and",
            "go",
            "left",
        ],
    },
    "Recognize speech without moving.": {
        "verbs": [
            {
                "verb": "Recognize",
                "description": "[V: Recognize] [ARG1: speech] [ARGM-MNR: without moving] .",
                "tags": ["B-V", "B-ARG1", "B-ARGM-MNR", "I-ARGM-MNR", "O"],
            },
            {
                "verb": "moving",
                "description": "Recognize speech without [V: moving] .",
                "tags": ["O", "O", "O", "B-V", "O"],
            },
        ],
        "words": ["Recognize", "speech", "without", "moving", "."],
    },
}


class MockSRL:
    def __init__(self):
        self.q = []
        self.used = set()

    def get(self):
        return SRL_CACHE[self.q.pop(0)]

    def put(self, s):
        assert s not in self.used
        self.used.add(s)
        self.q.append(s)


@pytest.fixture(scope="module", autouse=True)
def init():
    shared.init(srl_predictor_path=None)
    srl = MockSRL()
    shared.srl = srl
    shared.srl_put = srl.put
    shared.srl_get = srl.get


def sorted_subsentence_nodes(g):
    return [
        node
        for node, _ in sorted(
            nx.get_node_attributes(g, "idx").items(), key=itemgetter(1)
        )
    ]


def cmp_parsed_nodes(nodes, expected):
    assert len(nodes) == len(expected)
    for node, expected_str in zip(nodes, expected):
        assert str(node.parsed) == expected_str


def cmp_adj(g, n1, expected_len, *expected):
    adj = g.adj[n1]
    assert len(adj) == expected_len
    for n2, label in zip(expected[::2], expected[1::2]):
        assert "".join(t.text_with_ws for t in adj[n2]["label"]).strip() == label


def test_arm_motion_with_corref():
    doc = shared.spacy("Open your left hand and then extend it while saying hello.")
    result = process_document(doc)

    assert len(result) == 1

    g = result[0]
    assert len(g) == 4

    nodes = sorted_subsentence_nodes(g)
    cmp_parsed_nodes(
        nodes,
        [
            "ArmMotion(armMotion=OPEN,arm=LEFT)",  # TODO: order
            "ArmMotion(armMotion=EXTEND,arm=LEFT)",
            "Talk(text=Hello)",
        ],
    )

    n1, n2, n3 = nodes
    cmp_adj(g, n1, 1, n2, "and then")
    cmp_adj(g, n2, 3, n3, "while", None, ".")
    cmp_adj(g, n3, 1)


def test_without_moving():
    doc = shared.spacy("Recognize speech without moving.")
    result = process_document(doc)

    assert len(result) == 1

    g = result[0]
    assert len(g) == 3

    nodes = sorted_subsentence_nodes(g)
    cmp_parsed_nodes(nodes, ["Listen()", "BodyMotion()"])

    n1, n2 = nodes
    cmp_adj(g, n1, 2, n2, "without", None, ".")
    cmp_adj(g, n2, 1)


def test_onback():
    doc = shared.spacy("Say hello and lay on your back")
    result = process_document(doc)

    assert len(result) == 1

    g = result[0]
    assert len(g) == 3

    nodes = sorted_subsentence_nodes(g)
    cmp_parsed_nodes(nodes, ["Talk(text=Hello)", "BodyStance(stance=ONBACK)"])

    n1, n2 = nodes
    cmp_adj(g, n1, 1, n2, "and")
    cmp_adj(g, n2, 2, None, "")


def test_multiple_leds():
    doc = shared.spacy("Turn on the leds of your chest and legs and go left")
    result = process_document(doc)

    assert len(result) == 1

    g = result[0]
    assert len(g) == 3

    nodes = sorted_subsentence_nodes(g)
    # TODO: leds should be in a list/set type
    cmp_parsed_nodes(
        nodes,
        ["TurnLedOn(leds=CHEST_LEDS,leds=FEET_LEDS)", "BodyMotion(direction=LEFT)"],
    )

    n1, n2 = nodes
    cmp_adj(g, n1, 1, n2, "and")
    cmp_adj(g, n2, 2, None, "")


def test_all_used():
    keys = set(SRL_CACHE.keys())
    # No duplicates
    assert len(keys) == len(SRL_CACHE)

    call_args = list(shared.srl.used)

    while keys:
        s = keys.pop()
        call_args.remove(s)

    assert len(call_args) == 0
