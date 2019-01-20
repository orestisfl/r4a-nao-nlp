# TODO: docstrings
# vim:ts=4:sw=4:expandtab:fo-=t
from functools import reduce
from itertools import combinations, permutations
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Set, Tuple

from r4a_nao_nlp import logging
from r4a_nao_nlp.engines import parsed_score, shared

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import JsonDict, Span, Token

logger = logging.get_logger(__name__)


class SubSentence:
    def __init__(self, tags: List[str], sent: "Span"):
        logger.debug("Creating subsentence %d from tags %s", id(self), tags)

        self.tags = tags
        self.sent = sent

        self.verb: "Span" = None
        # XXX: these can be simple lists instead of dictionaries
        self.args: Dict[str, "Span"] = {}
        self.argms: Dict[str, "Span"] = {}
        self._parse_tags(tags)
        assert self.verb

        self.modifiers: Dict["SubSentence", Tuple["Span", "Span"]] = {}
        self.modifying: Dict["SubSentence", Tuple["Span", "Span"]] = {}
        self.compatible: Set["SubSentence"] = set()
        self.parsed: Optional["JsonDict"] = None

        logger.debug(
            "Attributes: verb: %s, args: %s, argms: %s",
            self.verb,
            self.args,
            self.argms,
        )

    def _parse_tags(self, tags: List[str]) -> None:
        start = 0
        end = None
        name = ""
        for idx, tag in enumerate(tags):
            if tag.startswith("B-"):
                self._save_tag(name, start, end)

                start = end = idx
                name = tag[2:]
            elif tag.startswith("I-"):
                assert name == tag[2:]
                end = idx
            else:
                assert tag == "O"
                self._save_tag(name, start, end)

                end = None
        self._save_tag(name, start, end)

    def _save_tag(self, tag: str, start: int, end: Optional[int]) -> None:
        if end is None:
            return

        span = self.sent[start : end + 1]
        if tag.startswith("V"):
            assert self.verb is None
            self.verb = span
        elif tag.startswith("ARGM-"):
            # TODO: check for existing tag
            self.argms[tag] = span
        else:
            assert tag.startswith("ARG")
            self.args[tag] = span

    def relate(self, other: "SubSentence") -> None:
        if self.is_compatible(other):
            self.compatible.add(other)
        else:
            return
        for span in self.argms.values():
            inter = span_intersect(span, other.verb)
            if inter:
                self.modifiers[other] = other.modifying[self] = (inter, span)
                # Assuming only one possible intersection between modifiers and verb range.
                return

    def is_compatible(self, other: "SubSentence") -> bool:
        return not span_search(other.verb, *self.args.values()) and not span_search(
            self.verb, other.verb, *other.args.values()
        )

    def parse(self, others: Optional[List["SubSentence"]] = None) -> "JsonDict":
        tokens = list(
            filter(
                lambda x: self.include_token(
                    x,
                    [modifier for modifier in self.modifiers if modifier in others]
                    if others is not None
                    else [],
                ),
                self.sent,
            )
        )

        best_result: Optional[Tuple[float, "JsonDict"]] = None
        for s in self._coref_combinations(tokens):
            logger.debug("Parsing %d using '%s'", id(self), s)
            result = shared.parse(s)
            score = parsed_score(result)
            if best_result is None or score > best_result[0]:
                best_result = (score, result)

        assert best_result is not None
        # XXX: this is hacky, cached value is used in process_document
        self.parsed = best_result[1]
        return best_result[1]

    def _coref_combinations(self, tokens: List["Token"]) -> Set[str]:
        # TODO: explain like coref_resolved etc

        base_tokens = tuple(token.text_with_ws for token in tokens)  # TODO:rename

        result = {"".join(base_tokens)}
        if not self.sent.doc._.has("coref_clusters"):
            return result

        base_clusters = {
            cluster for token in tokens for cluster in token._.coref_clusters
        }

        corefs = []
        for cluster in base_clusters:
            for coref in cluster:
                if coref == cluster.main or str(coref) == str(cluster.main):
                    # No point replacing this
                    continue

                indices = [idx for idx, token in enumerate(tokens) if token in coref]
                if indices:
                    corefs.append((cluster, coref, indices))

        for r in range(1, len(corefs) + 1):
            for c in combinations(corefs, r):
                resolved = list(base_tokens)
                for (cluster, coref, overlap) in c:
                    msg = (
                        "Replacing '"
                        + "".join(resolved[x] for x in overlap)
                        + "' with '"
                    )

                    resolved[overlap[0]] = (
                        cluster.main.text + self.sent.doc[coref.end - 1].whitespace_
                    )
                    for idx in overlap[1:]:
                        resolved[idx] = ""

                    msg += resolved[overlap[0]] + "'"
                    logger.debug(msg)
                result.add("".join(resolved))
        return result

    def include_token(self, token: "Token", modifiers: List["SubSentence"]) -> bool:
        return (
            token in self.verb
            or any(token in span for span in self.args.values())
            or (
                any(token in span for span in self.argms.values())
                and not any(other.include_token(token, []) for other in modifiers)
            )
        )

    def text_connect(
        self, modifiers: List["SubSentence"], other: Optional["SubSentence"] = None
    ) -> List["Token"]:
        start = self.verb.end
        end = other.verb.start if other else None
        return [
            token
            for token in self.sent[start:end]
            # TODO: (Test) argms tokens should be printed if not in other.
            if not self.include_token(token, modifiers)
            and (other is None or not other.include_token(token, modifiers))
            and not any(mod.include_token(token, []) for mod in modifiers)
        ]

    def __str__(self):
        return " ".join(self.tags)

    # Used for sorting
    def __lt__(self, other: object):
        if not isinstance(other, SubSentence):
            return NotImplemented

        if self.verb.start == other.verb.start:
            return self.verb.end < other.verb.end
        else:
            return self.verb.start < other.verb.start


class Combination:
    def __init__(self, subsentences: Iterable[SubSentence]):
        self.subsentences: List[SubSentence] = sorted(subsentences)
        self.sent: "Span" = self.subsentences[0].sent
        self._compatible: Optional[bool] = None
        self._parsed: Optional[List["JsonDict"]] = None

        assert all(
            subsentence.sent == self.sent for subsentence in self.subsentences[1:]
        )

    @property
    def compatible(self) -> bool:
        if self._compatible is None:
            self._compatible = self._check_compatible()
        return self._compatible

    def _check_compatible(self) -> bool:
        for idx in range(1, len(self)):
            next_subsentence = self.subsentences[idx]
            for subsentence in self.subsentences[:idx]:
                if next_subsentence not in subsentence.compatible:
                    return False
        return True

    @property
    def parsed(self) -> List["JsonDict"]:
        if self._parsed is None:
            self._parsed = [
                subsentence.parse(others=self.subsentences) for subsentence in self
            ]
        return self._parsed

    def to_complex_list(self):  # TODO: return value typing
        modifiers = [
            subsentence
            for subsentence in self
            if any(other for other in subsentence.modifying if other in self)
        ]
        rest = [subsentence for subsentence in self if subsentence not in modifiers]
        result = []
        for idx, subsentence in enumerate(rest):
            # TODO common_modifiers = {key: value for key, value in subsentence.modifiers.items() if key in modifiers}
            if subsentence is rest[-1]:
                other_words = subsentence.text_connect(modifiers)
                result.append((subsentence, other_words, None, None))

                logger.debug(
                    "Last subsentence: '%s', with words after: '%s'",
                    subsentence,
                    other_words,
                )
            else:
                next_subsentence = rest[idx + 1]
                other_words = subsentence.text_connect(modifiers, next_subsentence)
                result.append((subsentence, other_words, next_subsentence, None))

                logger.debug(
                    "SubSentence '%s', with words after: '%s'", subsentence, other_words
                )
            for key, value in subsentence.modifiers.items():
                if key in modifiers:
                    (inter, argms) = value
                    words_before = self.sent.doc[argms.start : inter.start]
                    if not words_before:
                        words_before = None
                    words_after = self.sent.doc[inter.end + 1 : argms.end]
                    if not words_after:
                        words_after = None
                    result.append((subsentence, words_before, key, words_after))

                    logger.debug(
                        "Modifier subsentence '%s' with words '%s' and '%s'",
                        key,
                        words_before,
                        words_after,
                    )
        return result

    def __iter__(self):
        return iter(self.subsentences)

    def __bool__(self):
        return bool(self.subsentences)

    def __len__(self):
        return len(self.subsentences)

    def __str__(self):
        return ", ".join(str(s) for s in self)


def span_intersect(*spans: "Span") -> "Span":
    if spans:
        doc = spans[0].doc
        return reduce(
            lambda a, b: doc[max(a.start, b.start) : min(a.end, b.end)], spans
        )


def span_search(value: "Span", *spans: "Span") -> Optional["Span"]:
    for span2 in spans:
        inter = span_intersect(value, span2)
        if inter:
            return inter
    return None


def create_combinations(sent: "Span") -> List[Combination]:
    result = shared.srl(str(sent))
    logger.debug("SRL (descriptions): %s", [v["description"] for v in result["verbs"]])
    assert result["words"] == [str(token) for token in sent]

    all_tags: List[List[str]] = [verb["tags"] for verb in result["verbs"]]
    subsentences = create_subsentences(all_tags, sent)
    return list(create_combinations_from_subsentences(subsentences))


def create_subsentences(all_tags: List[List[str]], sent: "Span") -> List[SubSentence]:
    # TODO: configurable threshold
    subsentences = list(
        filter(
            lambda s: parsed_score(s.parse()) > 0.1,
            (SubSentence(tags, sent) for tags in all_tags),
        )
    )
    for a, b in permutations(subsentences, r=2):
        a.relate(b)
    return subsentences


def create_combinations_from_subsentences(
    subsentences: List[SubSentence]
) -> Iterable[Combination]:
    contains = {subsentence: False for subsentence in subsentences}
    for r in range(len(subsentences), 0, -1):
        for combination_tuple in combinations(subsentences, r=r):
            if all(contains[subsentence] for subsentence in combination_tuple):
                continue

            combination = Combination(combination_tuple)
            if combination.compatible:
                yield combination

                for subsentence in combination:
                    contains[subsentence] = True
                if all(contains.values()):
                    # XXX: do we return too early here?
                    return
