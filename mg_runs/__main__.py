import argparse

from . import m2023_07_30 as current_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batchsize", type=int, required=True, dest="batch_size")

    parser.add_argument("--beta", type=float)

    parser.add_argument("--seed", type=int)

    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    args = {key: val for key, val in vars(args).items() if val is not None}

    print(f"Starting run {current_run.__name__} with args {args}")

    debug_info = current_run.run(**args)
