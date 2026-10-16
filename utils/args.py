import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    if v.lower() in ("no", "false", "f", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.ini",
        help="Path to the config file"
    )

    parser.add_argument(
        "--load_last_optimizer",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Load last optimizer state (default: True). Use true/false."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )


    parser.add_argument(
        "--catch_exceptions",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Wrap the execution with try and except (default: True). Use true/false."
    )

    parser.add_argument(
        "--test_mode",
        type=str2bool,
        nargs="?",
        const=False,
        default=False,
        help="Wrap the execution with try and except (default: True). Use true/false."
    )
    return parser.parse_args()