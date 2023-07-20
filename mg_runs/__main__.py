import argparse

from . import m2023_06_18 as current_run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--beta", type=float, default=None)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batchsize", type=int)

    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    debug = args.debug

    batch_size = args.batchsize

    if batch_size is None:
        raise Exception("batch size not set")

    beta = args.beta

    seed = args.seed

    print(f"Starting run {current_run.__name__} with batch size {batch_size}, seed {seed}, and beta {beta}" + " (debug)" if debug else "")

    current_run.run(batch_size, seed=seed, beta=beta, debug=debug)
