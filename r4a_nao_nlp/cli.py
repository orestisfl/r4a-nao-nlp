# TODO: docstrings
from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, List

from r4a_nao_nlp import core_nlp, subsentence, utils
from r4a_nao_nlp.engines import shared

if TYPE_CHECKING:
    from r4a_nao_nlp.typing import SnipsResult, Graph

logger = utils.create_logger(__name__)


def main(argv: List[str]) -> int:
    parse_command_line(argv[1:])

    shared.init()

    while True:
        try:
            line = input()
            if not line:
                continue
        except EOFError:
            return 0
        try:
            with open(line) as f:
                text = f.read()
        except OSError:
            logger.exception("Failed to load %s", line)
        else:
            process_document(text)


def parse_command_line(argv: List[str]) -> None:
    parser = utils.ArgumentParser()
    parser.add_ecore_root_argument("-r", "--ecore-path")
    parser.parse_args(argv)


@utils.timed
def process_document(s: str) -> Graph:
    # TODO: find a better way to deal with problems with whitespace.
    # Eliminate newlines and multiple whitespace.
    s = " ".join(s.split())

    s, replacements = core_nlp.replace_quotes(s)
    threads = core_nlp.CorefThreads(s, ("statistical",))
    doc = shared.spacy(s)
    for sent in doc.sents:
        shared.srl_put(str(sent))
    core_nlp.doc_mark_quotes(doc, replacements)
    core_nlp.doc_enhance_corefs(doc, threads.join())

    g = None
    for sent in doc.sents:
        logger.debug("Processing sent: %s", sent)

        srl_result = shared.srl_get()

        combinations = subsentence.create_combinations(sent, srl_result)
        logger.debug("Final combinations: %s", ", ".join(str(c) for c in combinations))
        if combinations:
            max_idx, max_score = max(
                enumerate(calc_score(c.parsed) for c in combinations), key=itemgetter(1)
            )
            max_combination = combinations[max_idx]

        simple = shared.parse(str(sent))
        if (
            not combinations
            or not max_score
            or (
                max_score < simple.score
                # Prefer a complex result with multiple intents
                and len(max_combination) == 1
                # If the result is the same anyway, don't use the full sentence
                and str(max_combination.parsed[0]) != str(simple)
            )
        ):
            logger.debug("Preferring full sentence")
            if g is None:
                from r4a_nao_nlp.graph import Graph

                g = Graph()
            else:
                g.sent_idx += 1

            if not simple:
                logger.warn("Could not parse intent from sentence: %s", sent)
                g.add_edge(g.prev_end, str(sent))
                continue

            node = subsentence.SubSentence(["B-V"] + (len(sent) - 1) * ["I-V"], sent)
            node.parsed = simple
            g.add_node(node, idx=0, idx_main=0)
            g.add_edge(node, "")
        else:
            g = max_combination.to_graph(g)

    return g


def calc_score(clauses: List[SnipsResult]) -> float:
    assert clauses

    # XXX: alternative scoring strategies
    # # product
    # return reduce(operator.mul, (clause.score for clause in clauses))
    # # min
    # return min((intent_clause(clause) for clause in clauses))
    # avg
    return sum((clause.score for clause in clauses)) / len(clauses)


if __name__ == "__main__":
    from r4a_nao_nlp import __main__

    __main__.entry_point(__spec__.name)

# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
