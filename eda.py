from typing import Tuple

import pandas as pd


def print_md(DIMs: Tuple) -> None:
    print(
        "|Server Arch.|Mean(Serial)|End(Serial)|Mean(Concurrency)|End(Concurrency)|Mean(Random)|End(Random)|"
    )
    print("|:-:|:-:|:-:|:-:|:-:|:-:|:-:|")
    for DIM in DIMs:
        res = []
        for tmp in ("SERIAL", "CONCURRENCY", "RANDOM"):
            res.append(
                f"{EACH[(EACH.iloc[:, 0] == tmp) & (EACH.iloc[:, 1].str.contains(DIM))].iloc[:, 2].mean():.3f}"
            )
            res.append(
                f"{TOTAL[(TOTAL.iloc[:, 0] == tmp) & (TOTAL.iloc[:, 1].str.contains(DIM))].iloc[:, 2].mean():.3f}"
            )
        print(f"|{DIM}|" + "|".join(res) + "|")
    print()


if __name__ == "__main__":
    EACH = pd.read_csv("data/EACH.csv", header=None)
    TOTAL = pd.read_csv("data/TOTAL.csv", header=None)

    EACH = EACH[~EACH.iloc[:, 1].str.contains("1-5")]
    TOTAL = TOTAL[~TOTAL.iloc[:, 1].str.contains("1-5")]

    print_md(("Sync", "Async", "Ensemble"))
    print_md(("Rep=1", "Rep=5"))
