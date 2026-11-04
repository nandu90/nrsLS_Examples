"""Plot single-phase versus bubbly-channel post-processing results."""

import argparse
import os

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", metavar="PATH",
                        help=("spectra.npz files or directories containing "
                              "spectra.npz"))
    parser.add_argument("--labels", nargs="+",
                        help="one plot label per --input case")
    # Retain the original two-case interface for existing workflows.
    parser.add_argument("--single-spectra")
    parser.add_argument("--two-spectra")
    parser.add_argument("--single-statistics")
    parser.add_argument("--two-statistics")
    parser.add_argument("--output", default="comparisonPlots")
    parser.add_argument("--target-yplus", type=float, nargs="*",
                        default=[12.0, 280.0])
    parser.add_argument("--relative-spectrum-floor", type=float,
                        default=1.0e-5,
                        help="omit plotted values below this fraction of peak")
    return parser.parse_args()


def spectra_filename(path):
    """Resolve either a spectra directory or an explicit NPZ filename."""
    filename = os.path.join(path, "spectra.npz") if os.path.isdir(path) else path
    if not os.path.isfile(filename):
        raise FileNotFoundError("Spectra file not found: %s" % filename)
    return filename


def matching_half_indices(y, yplus, target):
    """Select matching planes from the lower and upper channel halves."""
    selected = []
    for mask in (y <= 0.0, y >= 0.0):
        candidates = np.flatnonzero(mask)
        selected.append(candidates[np.argmin(np.abs(yplus[candidates]
                                               - target))])
    return np.unique(selected)


def folded(y, yplus, value):
    symmetric = 0.5*(value + value[::-1])
    upper = y >= -1.0e-12
    order = np.argsort(yplus[upper])
    return yplus[upper][order], symmetric[upper][order]


def wavenumber_spacing(wave):
    """Return the constant spacing of a uniformly sampled FFT grid."""
    if wave.size < 2:
        raise ValueError("At least two wavenumbers are required")
    spacing = float(wave[1] - wave[0])
    if spacing <= 0.0 or not np.allclose(np.diff(wave), spacing):
        raise ValueError("Expected uniformly spaced increasing wavenumbers")
    return spacing


def resolved(values, relative_floor):
    """Mask nonpositive values and the negligible numerical noise floor."""
    valid = np.isfinite(values) & (values > 0.0)
    if not np.any(valid):
        return valid
    return valid & (values >= relative_floor*np.max(values[valid]))


def main():
    args = arguments()
    if not 0.0 <= args.relative_spectrum_floor < 1.0:
        raise ValueError("relative-spectrum-floor must be in [0, 1)")
    os.makedirs(args.output, exist_ok=True)
    if args.input:
        if args.single_spectra or args.two_spectra:
            raise ValueError("Use either --input or the legacy spectra options")
        filenames = [spectra_filename(path) for path in args.input]
        labels = args.labels or [os.path.basename(os.path.dirname(name))
                                 for name in filenames]
    else:
        if not args.single_spectra or not args.two_spectra:
            raise ValueError("Supply --input, or both legacy spectra options")
        filenames = [spectra_filename(args.single_spectra),
                     spectra_filename(args.two_spectra)]
        labels = args.labels or ["single phase", "bubbly"]
    if len(labels) != len(filenames):
        raise ValueError("--labels must contain one label per input")
    spectra = [np.load(name) for name in filenames]
    styles = ["-", "--", ":", "-."]
    component_names = (r"$u'$", r"$v'$", r"$w'$")
    component_colors = ("#0072B2", "#D55E00", "#009E73")

    if bool(args.single_statistics) != bool(args.two_statistics):
        raise ValueError("Supply both statistics files or neither")
    if args.single_statistics:
        sp = np.load(args.single_statistics)
        tp = np.load(args.two_statistics)
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
        for data, label, style in ((sp, labels[0], "-"),
                                   (tp, labels[1], "--")):
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
        fig.savefig(os.path.join(args.output, "statisticsComparison.png"),
                    dpi=200)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 5.4))
        ax.plot([0.0, 1.0, 0.5, 0.0],
                [0.0, 0.0, np.sqrt(3.0)/2.0, 0.0], "k-")
        ax.plot(sp["liquidBarycentricX"], sp["liquidBarycentricY"],
                color="#555555", label=labels[0])
        ax.plot(tp["liquidBarycentricX"], tp["liquidBarycentricY"],
                color="#0072B2", label=labels[1])
        ax.text(1.02, 0.0, "1C", va="center")
        ax.text(-0.08, 0.0, "2C", va="center")
        ax.text(0.5, np.sqrt(3.0)/2.0 + 0.035, "3C", ha="center")
        ax.set_aspect("equal")
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.05, 0.95)
        ax.set_axis_off()
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(args.output, "barycentricComparison.png"),
                    dpi=200)
        plt.close(fig)

    for target in args.target_yplus:
        tag = ("%g" % target).replace(".", "p")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        actual = []
        for case, (data, label) in enumerate(zip(spectra, labels)):
            indices = matching_half_indices(
                data["y"], data["yplus"], target)
            actual.append(float(np.mean(data["yplus"][indices])))
            for ax, key_spectrum, key_wave, direction in (
                    (axes[0], "Ex", "kx", "x"),
                    (axes[1], "Ez", "kz", "z")):
                wave = data[key_wave]
                positive = wave > 0.0
                delta_wave = wavenumber_spacing(wave)
                for component in range(3):
                    spectrum = np.mean(
                        data[key_spectrum][indices, :, component], axis=0)
                    variance = np.mean(data["variance"][indices, component])
                    premultiplied = (
                        wave[positive]*spectrum[positive]
                        /(delta_wave*max(variance, 1.0e-30)))
                    valid = resolved(
                        premultiplied, args.relative_spectrum_floor)
                    ax.loglog(wave[positive][valid], premultiplied[valid],
                              color=component_colors[component],
                              linestyle=styles[case % len(styles)])
                ax.set_xlabel(r"$k_%s\delta$" % direction)
                ax.set_ylabel(
                    r"$k_%s E_{ii}/\langle u_i'^2\rangle$" % direction)
                ax.grid(True, which="both", alpha=0.35)
        component_handles = [
            Line2D([0], [0], color=color, linestyle="-", label=name)
            for name, color in zip(component_names, component_colors)]
        case_handles = [
            Line2D([0], [0], color="black",
                   linestyle=styles[case % len(styles)], label=label)
            for case, label in enumerate(labels)]
        axes[0].legend(handles=component_handles + case_handles)
        fig.suptitle(r"$y^+\approx %g$" % np.mean(actual))
        fig.tight_layout()
        fig.savefig(os.path.join(
            args.output, "premultipliedComparison_yplus_%s.png" % tag),
            dpi=200)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        actual = []
        for case, (data, label) in enumerate(zip(spectra, labels)):
            indices = matching_half_indices(data["y"], data["yplus"], target)
            actual.append(float(np.mean(data["yplus"][indices])))
            for ax, key_spectrum, key_wave, direction in (
                    (axes[0], "Ex", "kx", "x"),
                    (axes[1], "Ez", "kz", "z")):
                wave = data[key_wave]
                modal = np.mean(data[key_spectrum][indices], axis=0)
                density = (0.5*np.sum(modal, axis=1)
                           / wavenumber_spacing(wave))
                positive = wave > 0.0
                valid = resolved(
                    density[positive], args.relative_spectrum_floor)
                ax.loglog(wave[positive][valid], density[positive][valid],
                          color="#0072B2",
                          linestyle=styles[case % len(styles)], label=label)
                ax.set_xlabel(r"$k_%s\delta$" % direction)
                ax.set_ylabel(r"$E_{k,%s}$" % direction)
                ax.grid(True, which="both", alpha=0.35)
        axes[0].legend()
        fig.suptitle(r"TKE spectra, $y^+\approx %g$" % np.mean(actual))
        fig.tight_layout()
        fig.savefig(os.path.join(
            args.output, "tkeComparison_yplus_%s.png" % tag), dpi=200)
        plt.close(fig)

    print("Wrote comparison plots to %s" % args.output)


if __name__ == "__main__":
    main()
