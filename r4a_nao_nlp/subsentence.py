# TODO: docstrings
from __future__ import annotations

from contextlib import suppress
from functools import reduce
from itertools import chain, permutations
from logging import DEBUG, WARN
from typing import TYPE_CHECKING, Container, Dict, Iterable, List, Optional, Set, Tuple

from r4a_nao_nlp import utils
from r4a_nao_nlp.engines import SnipsResult, shared

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import JsonDict, Span, Token, Graph
    from r4a_nao_nlp.utils import before_37

    if before_37():
        SubSentence = "SubSentence"
        Combination = "Combination"

logger = utils.create_logger(__name__)


class SubSentence:
    def __init__(self, tags: List[str], sent: Span):
        logger.debug("Creating subsentence %d from tags %s", id(self), tags)

        self.tags = tags
        self.sent = sent

        self.verb: Span = None
        # XXX: these can be simple lists instead of dictionaries
        self.args: Dict[str, Span] = {}
        self.argms: Dict[str, Span] = {}
        self._parse_tags(tags)
        assert self.verb
        # XXX: maybe keep in self.verb?
        # http://universaldependencies.org/docs/en/dep/compound-prt.html
        self.particles: Set[Token] = set(
            child
            for token in self.verb
            for child in token.children
            if child.dep_ == "prt"
        )

        self.modifiers: Dict[SubSentence, Tuple[Span, Span]] = {}
        self.modifying: Dict[SubSentence, Tuple[Span, Span]] = {}
        self.compatible: Set[SubSentence] = set()
        self.parsed: Optional[SnipsResult] = None

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

    def relate(self, other: SubSentence) -> None:
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

    def is_compatible(self, other: SubSentence) -> bool:
        return not span_search(other.verb, *self.args.values()) and not span_search(
            self.verb, other.verb, *other.args.values()
        )

    def parse(self, others: Container[SubSentence] = ()) -> SnipsResult:
        modifiers: List[SubSentence] = [
            modifier for modifier in self.modifiers if modifier in others
        ]
        always: List[int] = []
        argms: Dict[Span, List[int]] = {}
        for idx, token in enumerate(self.sent):
            if token in self:
                always.append(idx)
            else:
                # We work with spans here since we don't want to partially include ARGM
                # spans - if any token of the corresponding ARGM span is included in any
                # of the specified modifiers, exclude the whole span.
                span = self._argm_with_token(token)
                if span and not any(
                    any(token in subsentence for token in span)
                    for subsentence in modifiers
                ):
                    indices = argms.get(span)
                    if indices:
                        indices.append(idx)
                    else:
                        argms[span] = [idx]

        # Shortcut that uses the 'always' list and the given combination of the values
        # in the argms dict.
        def parse(indices: Iterable[List[int]]) -> SnipsResult:
            return self._parse_from_token_indices(
                list(chain(always, chain.from_iterable(indices)))
            )

        self.parsed = parse(argms.values())
        self.used_argms = tuple(argms.keys())
        expected_result = str(self.parsed) if self.parsed else None
        # Iterate in reversed order to prefer using less ARGMs which could happen with a
        # perfect score of 1.0 (Snips' DeterministicIntentParser returns 1.0 when it
        # succeeds).
        for c in reversed(utils.PowerSet(argms, r_stop=-1)):
            c = tuple(c)
            current_max = parse(argms[key] for key in c)
            if current_max and (
                # Preserve the same parse result with any amount of modifiers so that we
                # don't end up using a simpler request. If the original intent was None,
                # skip this check to prefer ending up with any kind of non-None result.
                not expected_result
                or (self.parsed <= current_max and str(current_max) == expected_result)
            ):
                self.parsed = current_max
                self.used_argms = c
        return self.parsed

    def _parse_from_token_indices(self, indices: Container[int]) -> SnipsResult:
        tokens = [token for idx, token in enumerate(self.sent) if idx in indices]

        logger.debug("Parsing using tokens: %s", tokens)
        return max(
            (
                (n_coref_replacements, shared.parse_tokens(tokens_to_parse))
                for n_coref_replacements, tokens_to_parse in self._coref_combinations(
                    tokens
                )
            ),
            # Score results with more coref replacements higher.
            key=lambda v: (v[0] + 1) * float(v[1]),
        )[1]

    def _coref_combinations(self, tokens: List[Token]) -> Set[Tuple[int, List[Token]]]:
        # TODO: explain like coref_resolved etc

        result = {(0, tuple(tokens))}

        # We also need to check if .coref_clusters is empty because of
        # https://github.com/huggingface/neuralcoref/issues/58.
        if (
            not self.sent.doc._.has("coref_clusters")
            or not self.sent.doc._.coref_clusters
        ):
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

        for c in utils.PowerSet(corefs, r_start=1):
            resolved: List[Iterable[Token]] = [[token] for token in tokens]
            for (cluster, coref, overlap) in c:
                resolved[overlap[0]] = cluster.main
                for idx in overlap[1:]:
                    resolved[idx] = []

            result.add((len(c), tuple(chain.from_iterable(resolved))))
        return result

    def _argm_with_token(self, token: Token) -> Optional[Span]:
        for span in self.argms.values():
            if token in span:
                # assume no overlap in self's ARGMs
                return span
        return None

    def __contains__(self, item: object) -> bool:
        from spacy.tokens.token import Token

        if isinstance(item, Iterable):
            return all(token in self for token in item)
        if not isinstance(item, Token):
            raise TypeError(
                f"Argument 'item' has incorrect type: expected {Token}, got {type(item)}"
            )

        return (
            item in self.verb
            or item in self.particles
            or any(item in span for span in self.args.values())
        )

    def __str__(self):
        return " ".join(self.tags)

    # Used for sorting
    @utils.other_isinstance
    def __lt__(self, other: object):
        if self.verb.start == other.verb.start:
            return self.verb.end < other.verb.end
        else:
            return self.verb.start < other.verb.start


class Combination:
    def __init__(self, subsentences: Iterable[SubSentence]):
        self.subsentences: List[SubSentence] = sorted(subsentences)
        self.sent: Span = self.subsentences[0].sent
        self._compatible: Optional[bool] = None
        self._parsed: Optional[List[SnipsResult]] = None

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
    def parsed(self) -> List[SnipsResult]:
        if self._parsed is None:
            self._parsed = [
                subsentence.parse(others=self.subsentences) for subsentence in self
            ]
        return self._parsed

    def to_graph(self, graph: Optional[Graph] = None) -> Graph:
        if graph is None:
            from r4a_nao_nlp.graph import Graph

            graph = Graph()
        else:
            graph.sent_idx += 1

        modifiers = set(
            subsentence
            for subsentence in self
            if any(other for other in subsentence.modifying if other in self)
        )
        rest = [subsentence for subsentence in self if subsentence not in modifiers]

        connecting_tokens = [
            token
            for token in self.sent
            if not any(
                token in subsentence
                or any(token in span for span in subsentence.used_argms)
                for subsentence in self
            )
        ]
        logger.debug("Connecting tokens: %s", connecting_tokens)

        # Add all subsentences to the graph, index them using their original order.
        for idx, subsentence in enumerate(self):
            graph.add_node(subsentence, idx=idx)

        for subsentence in self:
            for key in modifiers.intersection(subsentence.modifiers.keys()):
                (inter, argm) = subsentence.modifiers[key]
                words_before = [
                    token
                    for token in self.sent.doc[argm.start : inter.start]
                    if token not in key
                ]
                words_after = [
                    token
                    for token in self.sent.doc[inter.end + 1 : argm.end]
                    if token not in key
                ]
                words = words_before or words_after
                graph.add_edge(key, words or None, subsentence)

                for token in words:
                    with suppress(ValueError):
                        connecting_tokens.remove(token)
                        logger.debug("connecting_tokens: removed %s", token)

                problem = words_before and words_after
                logger.log(
                    WARN if problem else DEBUG,
                    "Modifier subsentence '%s' with words '%s' and '%s'%s",
                    key,
                    words_before,
                    words_after,
                    ", skipping words_after" if problem else "",
                )

        graph.connect_prev(
            rest[0],
            [token for token in connecting_tokens if token.i < rest[0].verb.start],
        )

        for idx, subsentence in enumerate(rest):
            if subsentence is rest[-1]:
                next_subsentence = None
                end = self.sent[-1].i + 1
            else:
                next_subsentence = rest[idx + 1]
                end = next_subsentence.verb.start

            other_words = [
                token
                for token in connecting_tokens
                if subsentence.verb.end <= token.i < end
            ]
            graph.nodes[subsentence]["idx_main"] = idx
            graph.add_edge(subsentence, other_words, next_subsentence)

            logger.debug(
                "%ssubsentence: '%s', with words after: '%s'",
                "Last " if next_subsentence is None else "",
                subsentence,
                other_words,
            )

        return graph

    def __iter__(self):
        return iter(self.subsentences)

    def __bool__(self):
        return bool(self.subsentences)

    def __len__(self):
        return len(self.subsentences)

    def __str__(self):
        return ", ".join(str(s) for s in self)


def span_intersect(*spans: Span) -> Optional[Span]:
    if spans:
        doc = spans[0].doc
        result = reduce(
            lambda a, b: doc[max(a.start, b.start) : min(a.end, b.end)], spans
        )
        return result if result else None
    return None


def span_search(value: Span, *spans: Span) -> Optional[Span]:
    for span2 in spans:
        inter = span_intersect(value, span2)
        if inter:
            return inter
    return None


def create_combinations(sent: Span, srl_result: JsonDict) -> List[Combination]:
    logger.debug(
        "SRL (descriptions): %s",
        "\n\t- ".join(v["description"] for v in srl_result["verbs"]),
    )
    assert srl_result["words"] == [str(token) for token in sent]

    all_tags: List[List[str]] = [verb["tags"] for verb in srl_result["verbs"]]
    subsentences = create_subsentences(all_tags, sent)
    return list(create_combinations_from_subsentences(subsentences))


def create_subsentences(all_tags: List[List[str]], sent: Span) -> List[SubSentence]:
    # TODO: configurable threshold
    subsentences = list(
        filter(
            lambda s: s.parse().score > 0.1,
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
    for combination_tuple in reversed(utils.PowerSet(subsentences, r_start=1)):
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


# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
