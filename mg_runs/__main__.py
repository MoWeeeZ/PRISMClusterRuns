import argparse

from . import m2023_07_30 as current_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--batchsize", type=int, required=True, dest="batch_size")

    parser.add_argument("--beta", type=float)

    parser.add_argument("--seed", type=int)

    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    debug = args.debug

    batch_size = args.batchsize

    if batch_size is None:
        raise Exception("batch size not set")

    beta = args.beta

    seed = args.seed

    print(
        f"Starting run {current_run.__name__} with batch size {args.batch_size}, seed {args.seed}, and beta {args.beta}"
        + " (debug)"
        if args.debug
        else ""
    )

    args = {key: val for key, val in vars(args).items() if val is not None}

    debug_info = current_run.run(**args)
