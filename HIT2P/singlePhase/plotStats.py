from cycler import cycler
import glob
from mpi4py import MPI
import numpy as np
import os
import re
import time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def plotnow(fname, xlabel, ylabel, x, y, labels, ptype="line",
            linestyles=[], markers=[], ylim=[], xlim=[]):
    default_cycler = (
        cycler(color=["#0072B2", "#D55E00", "#009E73",
                      "#CC0000", "#990099"])*
        cycler(linestyle=["-"])*cycler(marker=[""])
    )
    plt.rc("lines", linewidth=1)
    plt.rc("axes", prop_cycle=default_cycler)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)

    if ylim != []:
        ax.set_ylim(ylim[0], ylim[1])

    if xlim != []:
        ax.set_xlim(xlim[0], xlim[1])

    for i in range(len(y)):
        if ptype == "line":
            ax.plot(x[i], y[i], label=labels[i],
                    linestyle=linestyles[i], marker=markers[i],
                    linewidth=2.0)
        elif ptype == "semilogx":
            ax.semilogx(x[i], y[i], label=labels[i],
                        linestyle=linestyles[i], marker=markers[i],
                        linewidth=2.0)
        elif ptype == "semilogy":
            ax.semilogy(x[i], y[i], label=labels[i],
                        linestyle=linestyles[i], marker=markers[i],
                        linewidth=2.0)
        else:
            ax.loglog(x[i], y[i], label=labels[i],
                      linestyle=linestyles[i], marker=markers[i],
                      linewidth=2.0)

    ax.grid()
    ax.legend(loc="best", fontsize=12)
    fig.savefig(fname + ".pdf", bbox_inches="tight", dpi=100)
    fig.savefig(fname + ".png", bbox_inches="tight", dpi=100)
    plt.close()

    return

def main():
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    fname = "HIT1p.dat"
    pattern = re.compile(
        r"HIT\s+t=\s*([-+0-9.eE]+)\s+"
        r"k=\s*([-+0-9.eE]+)\s+"
        r"eps=\s*([-+0-9.eE]+).*?"
        r"Re_lambda=\s*([-+0-9.eE]+)"
    )

    stats = {}
    with open(fname, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                t, k, eps, reLambda = map(float, match.groups())
                stats[t] = (k, eps, reLambda)

    if not stats:
        raise RuntimeError("No HIT statistics found in " + fname)

    t = np.array(sorted(stats))
    k = np.array([stats[ti][0] for ti in t])
    eps = np.array([stats[ti][1] for ti in t])
    reLambda = np.array([stats[ti][2] for ti in t])

    plotnow("ReLambda", r"$t$", r"$Re_\lambda$",
            [t, t], [reLambda, np.full_like(t, np.mean(reLambda))],
            [r"$Re_\lambda$", r"mean $= %.3f$" % np.mean(reLambda)],
            linestyles=["-", ":"], markers=["", ""])
    plotnow("tke", r"$t$", r"$k$",
            [t, t], [k, np.full_like(t, np.mean(k))],
            [r"$k$", r"mean $= %.3f$" % np.mean(k)],
            linestyles=["-", ":"], markers=["", ""])
    plotnow("dissipation", r"$t$", r"$\epsilon$",
            [t, t], [eps, np.full_like(t, np.mean(eps))],
            [r"$\epsilon$", r"mean $= %.3f$" % np.mean(eps)],
            linestyles=["-", ":"], markers=["", ""])

    print("Read %d records from %s" % (len(t), fname))

    return

if __name__ == "__main__":
    starttime = time.time()
    main()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("--- Code ran in %s seconds ---" %
              (time.time() - starttime))
