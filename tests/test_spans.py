import spacy

from r4a_nao_nlp.subsentence import span_intersect, span_search

doc = spacy.load("en_core_web_sm")("The quick brown fox jumps over the lazy dog")


def test_no_intersect():
    assert span_intersect() is None
    assert span_intersect(doc[1:3], doc[3:5]) is None
    assert span_intersect(doc[0:1], doc[6:]) is None


def test_simple_intersect():
    assert span_intersect(doc[1:3], doc[::]) == doc[1:3]
    assert span_intersect(doc[1:3], doc[2:3]) == doc[2:3]
    assert span_intersect(doc[::], doc[1:3]) == doc[1:3]
    assert span_intersect(doc[4:], doc[3:]) == doc[4:]
    assert span_intersect(doc[:3], doc[:2]) == doc[:2]


def test_multiple_intersect():
    assert span_intersect(doc[::], doc[::], doc[::]) == doc[::]
    assert span_intersect(doc[::], doc[5:4], doc[4:5]) is None
    assert span_intersect(doc[::], doc[1:3], doc[2:3]) == doc[2:3]
    assert span_intersect(doc[::], doc[1:5], doc[1:4], doc[1:3], doc[1:2]) == doc[1:2]
    assert span_intersect(doc[::], doc[4:5], doc[1:6], doc[1:4]) is None


def test_search():
    assert span_search(doc[::]) is None
    assert span_search(doc[::], doc[1:3], doc[4:5]) == doc[1:3]
    assert span_search(doc[1:3], doc[::], doc[0:5]) == doc[1:3]
    assert span_search(doc[1:3], doc[4:5], doc[6:7], doc[1:3]) == doc[1:3]
    assert span_search(doc[1:3], doc[4:5], doc[6:7], doc[0:2]) == doc[1:2]
    assert span_search(doc[1:3], doc[4:5], doc[6:7]) is None


# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
