# TODO: docstring
# vim:ts=4:sw=4:expandtab:fo-=t
def enter_cli_main():
    entry_point("r4a_nao_nlp.cli")


def enter_train_main():
    entry_point("r4a_nao_nlp.train")


def entry_point(main_module: str):
    import importlib
    import sys

    from r4a_nao_nlp import utils

    utils.init_logging()

    main = importlib.import_module(main_module).main
    sys.exit(main(sys.argv))


if __name__ == "__main__":
    enter_cli_main()
