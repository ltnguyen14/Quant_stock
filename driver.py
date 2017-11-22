from misc.arg_handler import arg_parser, InputHandler


def main():
    inputs = arg_parser()
    InputHandler(inputs)


if __name__ == "__main__":
    main()
