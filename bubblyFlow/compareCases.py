"""Plot single-phase versus bubbly-channel post-processing results."""

import argparse
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--single-statistics", required=True)
    parser.add_argument("--two-statistics", required=True)
    parser.add_argument("--single-spectra", required=True)
    parser.add_argument("--two-spectra", required=True)
    parser.add_argument("--output", default="comparisonPlots")
    parser.add_argument("--target-yplus", type=float, nargs="*",
                        default=[12.0, 280.0])
    return parser.parse_args()


def folded(y, yplus, value):
    symmetric = 0.5*(value + value[::-1])
    upper = y >= -1.0e-12
    order = np.argsort(yplus[upper])
    return yplus[upper][order], symmetric[upper][order]


def main():
    args = arguments()
    os.makedirs(args.output, exist_ok=True)
    sp = np.load(args.single_statistics)
    tp = np.load(args.two_statistics)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for data, label, style in ((sp, "single phase", "-"),
                               (tp, "bubbly", "--")):
        yp, k = folded(data["y"], data["yplus"], data["liquidTke"])
        axes[0].semilogx(yp, k, style, label=label)
        yp, b11 = folded(data["y"], data["yplus"],
                         data["liquidAnisotropy"][:, 0, 0])
        axes[1].semilogx(yp, b11, style, label=label)
        yp, flatness = folded(data["y"], data["yplus"],
                              data["liquidFlatness"])
        axes[2].semilogx(yp, flatness, style, label=label)

    axes[0].set_ylabel(r"$k_L$")
    axes[1].set_ylabel(r"$b_{11,L}$")
    axes[2].set_ylabel(r"$F_L$")
    for ax in axes:
        ax.set_xlabel(r"$y^+$")
        ax.grid(True, which="both", alpha=0.35)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "statisticsComparison.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot([0.0, 1.0, 0.5, 0.0],
            [0.0, 0.0, np.sqrt(3.0)/2.0, 0.0], "k-")
    ax.plot(sp["liquidBarycentricX"], sp["liquidBarycentricY"],
            color="#555555", label="single phase")
    ax.plot(tp["liquidBarycentricX"], tp["liquidBarycentricY"],
            color="#0072B2", label="bubbly")
    ax.text(1.02, 0.0, "1C", va="center")
    ax.text(-0.08, 0.0, "2C", va="center")
    ax.text(0.5, np.sqrt(3.0)/2.0 + 0.035, "3C", ha="center")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 0.95)
    ax.set_axis_off()
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.output, "barycentricComparison.png"), dpi=200)
    plt.close(fig)

    ss = np.load(args.single_spectra)
    ts = np.load(args.two_spectra)
    for target in args.target_yplus:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        for data, label, style in ((ss, "single phase", "-"),
                                   (ts, "bubbly", "--")):
            iy = int(np.argmin(np.abs(data["yplus"] - target)))
            re_tau = float(data["reTau"])
            for ax, key_spectrum, key_wave, direction in (
                    (axes[0], "Ex", "kx", "x"),
                    (axes[1], "Ez", "kz", "z")):
                wave = data[key_wave]
                spectrum = data[key_spectrum][iy, :, 0]
                positive = wave > 0.0
                wavelength_plus = 2.0*np.pi*re_tau/wave[positive]
                premultiplied = (wave[positive]*spectrum[positive]
                                 / max(data["variance"][iy, 0], 1.0e-30))
                order = np.argsort(wavelength_plus)
                ax.semilogx(wavelength_plus[order], premultiplied[order],
                            style, label=label)
                ax.set_xlabel(r"$\lambda_%s^+$" % direction)
                ax.set_ylabel(r"$k_%sE_{uu}/\langle u'^2\rangle$" % direction)
                ax.grid(True, which="both", alpha=0.35)
        axes[0].legend()
        fig.suptitle(r"$y^+\approx %g$" % target)
        fig.tight_layout()
        tag = ("%g" % target).replace(".", "p")
        fig.savefig(os.path.join(args.output,
                                 "spectraComparison_yplus_%s.png" % tag),
                    dpi=200)
        plt.close(fig)

    print("Wrote comparison plots to %s" % args.output)


if __name__ == "__main__":
    main()
