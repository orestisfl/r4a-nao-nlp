from string import ascii_lowercase
from typing import List, Optional
from unittest.mock import Mock

from r4a_nao_nlp import subsentence
from r4a_nao_nlp.engines import shared
from spacy.tokens.span import Span

# We only need spacy
shared.init(
    snips_path=None,
    srl_predictor_path=None,
    neural_coref_model=None,
    spacy_lang="en_core_web_sm",
)
shared.parse = Mock(return_value={"intent": {"probability": 1.0}})


def create_spacy_sent(length: int, s: Optional[str] = None) -> Span:
    """Return a fake spacy sentence of the given length."""
    if s is None:
        assert length is not None
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


def mock_srl(all_tags: List[List[str]], sent: Span) -> Mock:
    """Return a mock object to be used in place of actual semantic role labeling."""
    return Mock(
        return_value={
            "words": [str(token) for token in sent],
            "verbs": [{"tags": tags} for tags in all_tags],
        }
    )


def test_single_combination():
    all_tags = [
        ["B-V", "O", "O", "O", "B-ARGM-TMP", "I-ARGM-TMP"],
        ["O", "O", "B-V", "B-ARG1", "B-ARGM-TMP", "I-ARGM-TMP"],
        ["O", "O", "O", "O", "O", "B-V"],
    ]
    sent = create_spacy_sent(6)
    shared.srl = mock_srl(all_tags, sent)
    c = subsentence.create_combinations(sent)

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
    shared.srl = mock_srl(all_tags, sent)
    c = subsentence.create_combinations(sent)

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
    shared.srl = mock_srl(all_tags, sent)
    c = subsentence.create_combinations(sent)
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


def test_complex_list():
    all_tags = [
        ["B-V", "B-ARGM-TMP", "I-ARGM-TMP", "O", "O", "O"],
        ["O", "O", "B-V", "B-ARG1", "O", "O"],
        ["O", "O", "O", "B-ARG1", "B-V", "O"],
    ]
    sent = create_spacy_sent(6)
    shared.srl = mock_srl(all_tags, sent)
    c = subsentence.create_combinations(sent)

    assert len(c) == 1

    result = c[0].to_complex_list()
    assert len(result) == 3

    assert result[0][0].tags == all_tags[0]
    assert len(result[0][1]) == 0
    assert result[0][2].tags == all_tags[2]
    assert result[0][3] is None

    assert result[1][0].tags == all_tags[0]
    assert [t.text for t in result[1][1]] == [idx_to_word(1)]
    assert result[1][2].tags == all_tags[1]
    assert result[1][3] is None

    assert result[2][0].tags == all_tags[2]
    assert [t.text for t in result[2][1]] == ["."]
    assert result[2][2] is None
    assert result[2][3] is None


def test_complex_list_common_argms():
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
    shared.srl = mock_srl(all_tags, sent)
    c = subsentence.create_combinations(sent)
    assert len(c) == 1

    result = c[0].to_complex_list()
    assert len(result) == 4

    assert result[0][0].tags == all_tags[0]
    assert [t.text for t in result[0][1]] == ["and"]
    assert result[0][2].tags == all_tags[1]
    assert result[0][3] is None

    assert result[1][0].tags == all_tags[0]
    assert [t.text for t in result[1][1]] == ["while"]
    assert result[1][2].tags == all_tags[2]
    assert result[1][3] is None

    assert result[2][0].tags == all_tags[1]
    assert [t.text for t in result[2][1]] == ["."]
    assert result[2][2] is None
    assert result[2][3] is None

    assert result[3][0].tags == all_tags[1]
    assert [t.text for t in result[3][1]] == ["while"]
    assert result[3][2].tags == all_tags[2]
    assert result[3][3] is None
