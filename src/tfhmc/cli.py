import argparse
from pathlib import Path

from . import experiments
from .data import DATA_DIR, DATASETS


def main(argv=None):
    parser = argparse.ArgumentParser(prog="tfhmc")
    sub = parser.add_subparsers(dest="command", required=True)

    f = sub.add_parser("fit")
    f.add_argument("--datasets", default=",".join(DATASETS))
    f.add_argument("--data", type=Path, default=DATA_DIR)
    f.add_argument("--scores", type=Path, default=Path("scores"))
    f.add_argument("--device", default="auto")
    f.add_argument("--shard", default="0/1")

    e = sub.add_parser("evaluate")
    e.add_argument("--datasets", default=",".join(DATASETS))
    e.add_argument("--data", type=Path, default=DATA_DIR)
    e.add_argument("--scores", type=Path, default=Path("scores"))
    e.add_argument("--out", type=Path, default=Path("results"))

    g = sub.add_parser("lgbm")
    g.add_argument("--datasets", default=",".join(DATASETS))
    g.add_argument("--data", type=Path, default=DATA_DIR)
    g.add_argument("--out", type=Path, default=Path("results/lgbm.csv"))
    g.add_argument("--jobs", type=int, default=-1)

    r = sub.add_parser("reruns")
    r.add_argument("--chmcnn", type=Path, action="append", default=[])
    r.add_argument("--spur", type=Path, action="append", default=[])
    r.add_argument("--out", type=Path, default=Path("results/reruns.csv"))

    args = parser.parse_args(argv)
    if args.command == "fit":
        i, k = (int(v) for v in args.shard.split("/"))
        experiments.fit(args.datasets.split(","), args.scores, args.data, args.device, (i, k))
    elif args.command == "evaluate":
        experiments.evaluate(args.datasets.split(","), args.scores, args.out, args.data)
    elif args.command == "lgbm":
        experiments.lgbm(args.datasets.split(","), args.out, args.data, args.jobs)
    else:
        experiments.reruns(args.chmcnn, args.spur, args.out)


if __name__ == "__main__":
    main()
