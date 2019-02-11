# TODO: docstrings
# vim:ts=4:sw=4:expandtab:fo-=t
from __future__ import annotations

from typing import TYPE_CHECKING, List

from r4a_nao_nlp import subsentence, utils
from r4a_nao_nlp.engines import shared
from r4a_nao_nlp.graph import Graph

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import Doc
    from r4a_nao_nlp.engines import SnipsResult

logger = utils.create_logger(__name__)


def main(argv: List[str]) -> int:
    parse_command_line(argv[1:])

    shared.init()

    while True:
        try:
            line = input()
        except EOFError:
            return 0
        try:
            with open(line) as f:
                text = f.read()
        except OSError:
            logger.exception("Failed to load %s", line)
        else:
            process_document(shared.spacy(text))


def parse_command_line(argv: List[str]) -> None:
    parser = utils.ArgumentParser()
    parser.parse_args(argv)


def process_document(doc: Doc) -> None:
    for sent in doc.sents:
        logger.debug("Processing sent: %s", str(sent))

        # XXX: create all combinations for each sentence in background for speed-up
        combinations = subsentence.create_combinations(sent)
        logger.debug("Final combinations: %s", ", ".join(str(c) for c in combinations))
        if combinations:
            scores = [calc_score(c.parsed) for c in combinations]
            max_idx = max(enumerate(scores), key=lambda x: x[1])[0]

        simple = shared.parse(str(sent))
        # TODO: modifier? configurable?
        if not combinations or scores[max_idx] < 0.95 * simple.score:
            logger.debug("Prefering full sentence")
            g = Graph()
            g.add_node(
                subsentence.SubSentence(["B-V"] + (len(sent) - 1) * ["I-V"], sent)
            )
        else:
            g = combinations[max_idx].to_graph()
        g.plot(str(sent) + ("." if str(sent[-1]) != "." else "") + "pdf")


def calc_score(clauses: List[SnipsResult]) -> float:
    assert clauses

    # XXX: alternative scoring strategies
    # # product
    # return reduce(operator.mul, (clause.score for clause in clauses))
    # # min
    # return min((intent_clause(clause) for clause in clauses))
    # avg
    return sum((clause.score for clause in clauses)) / len(clauses)
