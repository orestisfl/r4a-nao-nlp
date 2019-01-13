# TODO: docstring
# vim:ts=4:sw=4:expandtab:fo-=t
from r4a_nao_nlp.cli import main

if __name__ == "__main__":
    import sys
    import logging

    logging.basicConfig()
    sys.exit(main(sys.argv))
