#!/usr/bin/env python
"""Stand-alone script that prints the output of the allennlp SRL model for the string
given in its stdin.

We can't use the multiprocessing module because it doesn't call atexit functions on
process shutdown:
https://stackoverflow.com/a/34507557/
https://github.com/python/cpython/blob/49fd6dd887df6ea18dbb1a3c0f599239ccd1cb42/Lib/multiprocessing/popen_fork.py#L75
But it is important for allennlp or else extracted archives are left in the $TMPDIR:
https://github.com/allenai/allennlp/blob/fefc439035df87e3d2484eb2f53ca921c4c2e2fe/allennlp/models/archival.py#L176-L178
Using os.fork() and allowing cleanups to be run like normal is dangerous because some
filesystem-related cleanups might be called twice.
"""
import json
import logging
import sys

# We don't use any loggers here, use basicConfig to mute allennlp's output.
logging.basicConfig(level=logging.WARN)


def main(args):
    r"""Load the predictor and run the loop that accepts input from stdin.

    Input should be delimited with \0\n.
    Output is delimited with \n.
    """
    if len(args) != 1:
        print("Please provide the model path", file=sys.stderr)
        sys.exit(1)
    path = args[0]

    from allennlp.predictors.predictor import Predictor

    predictor = Predictor.from_path(path)

    s = ""
    while True:
        line = sys.stdin.buffer.readline()
        if len(line) == 0:
            break
        if len(line) == 1:
            continue
        if line[-2] == 0:
            s += line[:-2:].decode()
            print(
                json.dumps(predictor.predict(s), indent=None, separators=(",", ":")),
                flush=True,
            )
            s = ""
        else:
            s += line.decode()


if __name__ == "__main__":
    main(sys.argv[1:])

# vim:ts=4:sw=4:expandtab:fo-=t:tw=88
