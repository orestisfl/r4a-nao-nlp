from spacy.util import compounding, minibatch

TRAIN_DATA = [
    ("Replay motion XXX", {"tags": ["VB", "NN", "NNP"]}),
    ("Turn right and then left", {"tags": ["VB", "RB", "CC", "RB", "RB"]}),
]

with nlp.disable_pipes(*(pipe for pipe in nlp.pipe_names if pipe != "tagger")):
    for _ in range(100):
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(
                texts,  # batch of texts
                annotations,  # batch of annotations
                drop=0.5,  # dropout - make it harder to memorise data
                # losses=losses,
            )
