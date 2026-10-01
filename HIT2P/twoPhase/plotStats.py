from cycler import cycler
import glob
from mpi4py import MPI
import numpy as np
import os
import time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def plotnow(fname, xlabel, ylabel, x, y, labels, ptype="line",
            linestyles=[], markers=[], ylim=[], xlim=[], showmean=False):
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
            line, = ax.plot(x[i], y[i], label=labels[i],
                            linestyle=linestyles[i], marker=markers[i],
                            linewidth=2.0)
        elif ptype == "semilogx":
            line, = ax.semilogx(x[i], y[i], label=labels[i],
                                linestyle=linestyles[i], marker=markers[i],
                                linewidth=2.0)
        elif ptype == "semilogy":
            line, = ax.semilogy(x[i], y[i], label=labels[i],
                                linestyle=linestyles[i], marker=markers[i],
                                linewidth=2.0)
        else:
            line, = ax.loglog(x[i], y[i], label=labels[i],
                              linestyle=linestyles[i], marker=markers[i],
                              linewidth=2.0)

        if showmean:
            mean = np.mean(y[i])
            ax.axhline(mean, color=line.get_color(), linestyle=":",
                       linewidth=2.0,
                       label=labels[i] + r" mean $= %.3g$" % mean)

    ax.grid()
    ax.legend(loc="best", fontsize=12)
    fig.savefig(fname + ".pdf", bbox_inches="tight", dpi=100)
    fig.savefig(fname + ".png", bbox_inches="tight", dpi=100)
    plt.close()

    return

def main():
    if MPI.COMM_WORLD.Get_rank() != 0:
        return

    fname = "hit2p.dat"
    data = np.loadtxt(fname, comments="#", ndmin=2)
    if data.shape[1] != 18:
        raise RuntimeError("Expected 18 columns in " + fname)

    # Keep the last entry if restart output contains a duplicate time.
    data = data[::-1]
    _, indices = np.unique(data[:, 0], return_index=True)
    data = data[indices]
    data = data[np.argsort(data[:, 0])]

    t = data[:, 0]
    alphaD, area, se = data[:, 1], data[:, 2], data[:, 3]
    kMix, epsMix = data[:, 4], data[:, 5]
    kC, kD = data[:, 6], data[:, 7]
    epsC, epsD = data[:, 8], data[:, 9]
    reLambdaC, reLambdaD = data[:, 10], data[:, 11]
    uAvgC = data[:, 12:15]
    uAvgD = data[:, 15:18]

    def makeplot(name, ylabel, values, labels):
        n = len(values)
        plotnow(name, r"$t$", ylabel, [t]*n, values, labels,
                linestyles=["-"]*n, markers=[""]*n, showmean=True)

    makeplot("volumeFraction", r"$\alpha_D$", [alphaD], [r"$\alpha_D$"])
    makeplot("interfaceArea", r"$S$", [area], [r"$S$"])
    makeplot("surfaceEnergy", r"$se$", [se], [r"$se$"])
    makeplot("tke", r"$k$", [kMix, kC, kD],
             [r"$k_{mix}$", r"$k_C$", r"$k_D$"])
    makeplot("dissipation", r"$\epsilon$", [epsMix, epsC, epsD],
             [r"$\epsilon_{mix}$", r"$\epsilon_C$", r"$\epsilon_D$"])
    makeplot("ReLambda", r"$Re_\lambda$", [reLambdaC, reLambdaD],
             [r"$Re_{\lambda,C}$", r"$Re_{\lambda,D}$"])
    makeplot("uAvgCarrier", r"$\langle u_i\rangle_C$",
             [uAvgC[:, 0], uAvgC[:, 1], uAvgC[:, 2]],
             [r"$\langle u\rangle_C$", r"$\langle v\rangle_C$",
              r"$\langle w\rangle_C$"])
    makeplot("uAvgDispersed", r"$\langle u_i\rangle_D$",
             [uAvgD[:, 0], uAvgD[:, 1], uAvgD[:, 2]],
             [r"$\langle u\rangle_D$", r"$\langle v\rangle_D$",
              r"$\langle w\rangle_D$"])

    print("Read %d records from %s" % (data.shape[0], fname))

    return

if __name__ == "__main__":
    starttime = time.time()
    main()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("--- Code ran in %s seconds ---" %
              (time.time() - starttime))
