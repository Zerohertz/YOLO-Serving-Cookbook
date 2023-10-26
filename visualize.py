from glob import glob

import numpy as np
from matplotlib import pyplot as plt

plt.rcParams["font.size"] = 20
plt.rcParams["font.family"] = "Times New Roman"


def read_data(FILE_NAME):
    with open(FILE_NAME, "r") as f:
        tmp = f.readlines()
    tmp = [float(t.strip()) for t in tmp]
    return tmp[:-1], tmp[-1], max(tmp[:-1])


if __name__ == "__main__":
    COMPARE = [
        "Sync&Rep=1",
        "Sync&Rep=5",
        "Async&Rep=1",
        "Async&Rep=1-5",
        "Async&Rep=5",
        "Ensemble&Rep=1",
        "Ensemble&Rep=5",
    ]
    COLOR = {
        "Sync&Rep=1": "#1f77b4",
        "Sync&Rep=5": "#ff7f0e",
        "Async&Rep=1": "#2ca02c",
        "Async&Rep=1-5": "#d62728",
        "Async&Rep=5": "#9467bd",
        "Ensemble&Rep=1": "#8c564b",
        "Ensemble&Rep=5": "#e377c2",
    }

    EACH, TOTAL = {}, {}
    BINS = {"SERIAL": 0, "CONCURRENCY": 0, "RANDOM": 0}
    for tmp in glob("data/*/*.txt"):
        each, total, bins = read_data(tmp)
        _, SERVER, CLIENT = tmp.split("/")
        CLIENT = CLIENT[:-6]
        if SERVER in COMPARE:
            BINS[CLIENT] = max(bins, BINS[CLIENT])
        if CLIENT in TOTAL.keys():
            if SERVER in TOTAL[CLIENT].keys():
                EACH[CLIENT][SERVER] += each
                TOTAL[CLIENT][SERVER].append(total)
            else:
                EACH[CLIENT][SERVER] = each
                TOTAL[CLIENT][SERVER] = [total]
        else:
            EACH[CLIENT] = {SERVER: each}
            TOTAL[CLIENT] = {SERVER: [total]}

    for k, v in EACH.items():
        fig = plt.figure(figsize=(15, 10))
        bins = np.linspace(0, BINS[k], 100)
        plt.grid(True)
        for kk in COMPARE:
            try:
                vv = v[kk]
                plt.hist(vv, bins=bins, color=COLOR[kk], label=kk, alpha=0.7, zorder=10)
                print(f"{k}\t{kk}\t{sum(vv) / len(vv):.2f}")
            except:
                pass
        plt.xlabel("API Response Time [Sec]")
        plt.ylabel("API Response Count")
        plt.legend()
        plt.title(f"CLIENT: {k}")
        plt.savefig(
            f"figures/EACH-{k}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
            transparent=False,
        )

    for k, v in TOTAL.items():
        idx = np.linspace(-0.4, 0.4, 10)
        bar_width = (idx[1] - idx[0]) * 0.95
        fig = plt.figure(figsize=(25, 10))
        plt.grid(True)
        xt = []
        i = 0
        for kk in COMPARE:
            try:
                vv = v[kk]
                plt.bar(i + idx, vv, bar_width, color=COLOR[kk], zorder=10)
                print(f"{k}\t{kk}\t{sum(vv) / len(vv):.2f}")
                xt.append(kk)
                i += 1
            except:
                pass
        plt.xlabel("Server Architecture")
        plt.ylabel("Total API Response Time [Sec]")
        plt.xticks(np.arange(i), xt, rotation=0)
        plt.title(f"CLIENT: {k}")
        plt.savefig(
            f"figures/TOTAL-{k}.png",
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.3,
            transparent=False,
        )
