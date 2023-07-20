import argparse

from . import m2023_06_18

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--temperature", type=float, default=None)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batchsize", type=int)

    parser.add_argument("--debug", action="store_true", default=False)

    args = parser.parse_args()

    debug = args.debug

    batch_size = args.batchsize

    if batch_size is None:
        raise Exception("batch size not set")

    temperature = args.temperature

    seed = args.seed

    m2023_06_18.run(batch_size, seed=seed, temperature=temperature, debug=debug)
