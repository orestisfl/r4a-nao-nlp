# TODO: docstrings
import logging
from typing import List

from r4a_nao_nlp import subsentence
from r4a_nao_nlp.engines import JsonDict, parsed_score, shared
from spacy.tokens.doc import Doc

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # TODO


def main() -> int:
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
        if not combinations or scores[max_idx] < parsed_score(simple):
            logger.debug("Prefering full sentence")
            print(snips_dict_to_str(simple))
            continue

        max_combination = combinations[max_idx]
        max_combination_complex = max_combination.to_complex_list()
        logger.debug(
            "Using combination %d - %s",
            max_idx,
            ",".join(str(s) for t in max_combination_complex for s in t),
        )
        for subsentence1, words1, subsentence2, words2 in max_combination_complex:
            # TODO: better printing
            result = "- " if words2 else ""
            result += snips_dict_to_str(subsentence1.parsed)
            if result is None:
                logger.error("No intend in result")
                continue
            if words1:
                result += " " + " ".join(str(token) for token in words1)
            if subsentence2 and words2:
                result += snips_dict_to_str(subsentence2.parsed)
            if words2:
                result += " " + " ".join(str(token) for token in words2)
            print(result)


def snips_dict_to_str(parsed: JsonDict) -> str:
    if parsed["intent"] is None:
        return None
    return "{intent}({args})".format(
        intent=parsed["intent"]["intentName"],
        args=",".join(
            "{slot}={value}".format(slot=slot["slotName"], value=slot_value_repr(slot))
            for slot in parsed["slots"]
        ),
    )


def slot_value_repr(slot: JsonDict) -> str:
    if "value" in slot["value"]:
        return slot["value"]["value"]
    else:
        return slot["rawValue"]
    # TODO snips/duration etc


def calc_score(clauses: List[JsonDict]) -> float:
    assert clauses

    # XXX: alternative scoring strategies
    # # product
    # return reduce(operator.mul, (parsed_score(clause) for clause in clauses))
    # # min
    # return min((intent_clause(clause) for clause in clauses))
    # avg
    return sum((parsed_score(clause) for clause in clauses)) / len(clauses)
