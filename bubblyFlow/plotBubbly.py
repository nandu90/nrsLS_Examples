"""Plot the time histories written to bubbly.dat."""

from cycler import cycler
import argparse
import os
import time

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


COLUMNS = (
    "t", "alphaG", "volumeG", "relativeVolumeError", "area",
    "interfaceThickness", "d32", "surfaceEnergy", "kL", "kG",
    "uLx", "uLy", "uLz", "uGx", "uGy", "uGz",
    "slipX", "slipY", "slipZ", "slipMagnitude",
)

def plotnow(fname, xlabel, ylabel, x, y, labels, ptype="line",
            linestyles=[], markers=[], colors=[], ylim=[], xlim=[],
            legend_fontsize=12, show_all_log_decades=False,
            quantity_legend_labels=None, case_legend_labels=None):
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
        color = colors[i] if colors != [] else None
        if ptype == "line":
            ax.plot(x[i], y[i], label=labels[i],
                    linestyle=linestyles[i], marker=markers[i],
                    color=color, linewidth=2.0)
        elif ptype == "semilogx":
            ax.semilogx(x[i], y[i], label=labels[i],
                        linestyle=linestyles[i], marker=markers[i],
                        color=color, linewidth=2.0)
        elif ptype == "semilogy":
            ax.semilogy(x[i], y[i], label=labels[i],
                        linestyle=linestyles[i], marker=markers[i],
                        color=color, linewidth=2.0)
        else:
            ax.loglog(x[i], y[i], label=labels[i],
                      linestyle=linestyles[i], marker=markers[i],
                      color=color, linewidth=2.0)

    if ptype == "semilogy" and show_all_log_decades:
        ax.yaxis.set_major_locator(
            mticker.LogLocator(base=10.0, subs=(1.0,), numticks=100))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10.0,
                               subs=np.arange(2.0, 10.0)*0.1,
                               numticks=100))
        ax.yaxis.set_minor_formatter(mticker.NullFormatter())
        ax.tick_params(axis="y", which="minor", length=3)

    ax.grid()
    if quantity_legend_labels is not None and case_legend_labels is not None:
        number_cases = len(case_legend_labels)
        quantity_handles = [
            Line2D([0], [0], color=colors[index*number_cases],
                   linestyle="-", linewidth=2.0, label=label)
            for index, label in enumerate(quantity_legend_labels)
        ]
        case_handles = [
            Line2D([0], [0], color="black", linestyle=linestyles[index],
                   linewidth=2.0, label=label)
            for index, label in enumerate(case_legend_labels)
        ]
        quantity_legend = ax.legend(
            handles=quantity_handles, loc="upper left",
            fontsize=legend_fontsize)
        ax.add_artist(quantity_legend)
        ax.legend(handles=case_handles, loc="upper right",
                  fontsize=legend_fontsize)
    else:
        ax.legend(loc="best", fontsize=legend_fontsize)
    fig.savefig(fname + ".pdf", bbox_inches="tight", dpi=100)
    fig.savefig(fname + ".png", bbox_inches="tight", dpi=100)
    plt.close()

    return


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+",
                        default=["p7/bubbly.dat", "p9/bubbly.dat",
                                 "h2_p7/bubbly.dat"],
                        help="Input monitor files")
    parser.add_argument("--labels", nargs="+",
                        default=["E,N7", "E,N9", "2E,N7"],
                        help="Labels corresponding to the input files")
    parser.add_argument("--output", default="bubblyPlots",
                        help="Output directory (default: bubblyPlots)")
    parser.add_argument("--t-min", type=float, default=None,
                        help="Discard samples before this time")
    parser.add_argument("--t-max", type=float, default=None,
                        help="Discard samples after this time")
    return parser.parse_args()


def read_bubbly(filename, t_min=None, t_max=None):
    values = np.loadtxt(filename, comments="#", ndmin=2)
    if values.shape[1] != len(COLUMNS):
        raise ValueError(
            "%s has %d columns; expected %d" %
            (filename, values.shape[1], len(COLUMNS)))

    finite = np.all(np.isfinite(values), axis=1)
    values = values[finite]
    if t_min is not None:
        values = values[values[:, 0] >= t_min]
    if t_max is not None:
        values = values[values[:, 0] <= t_max]
    if values.shape[0] == 0:
        raise RuntimeError("No samples remain after filtering")

    values = values[np.argsort(values[:, 0], kind="stable")]
    return {name: values[:, index]
            for index, name in enumerate(COLUMNS)}


def plot_bubbly(output_dir, datasets, case_labels):
    os.makedirs(output_dir, exist_ok=True)
    times = [data["t"] for data in datasets]
    case_linestyles = ["-", ":", "--", "-."]
    linestyles = [case_linestyles[i % len(case_linestyles)]
                  for i in range(len(datasets))]
    markers = [""]*len(datasets)
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC0000",
               "#990099"]
    case_colors = [palette[index % len(palette)]
                   for index in range(len(datasets))]

    def field_colors(number_fields):
        return [palette[field % len(palette)]
                for field in range(number_fields)
                for _ in datasets]

    def late_time_mean(data, field, mean_start):
        t = data["t"]
        if t[-1] <= mean_start:
            return None

        values = data[field]
        after = t > mean_start
        average_times = np.concatenate(([mean_start], t[after]))
        average_values = np.concatenate(
            ([np.interp(mean_start, t, values)], values[after]))
        return np.sum(
            0.5*(average_values[:-1] + average_values[1:]) *
            np.diff(average_times)
        )/(average_times[-1] - average_times[0])

    def plot_field(filename, field, ylabel, ptype="line"):
        plotnow(
            os.path.join(output_dir, filename),
            r"$t$", ylabel,
            times,
            [data[field] for data in datasets],
            case_labels,
            linestyles=linestyles,
            markers=markers,
            colors=case_colors,
            ptype=ptype,
        )

    mean_fields = (
        ("area", "A"),
        ("d32", "d32"),
        ("surfaceEnergy", "SE"),
        ("kL", "k_l"),
        ("kG", "k_g"),
        ("uLx", "<u_x>_l"),
        ("uGx", "<u_x>_g"),
        ("slipX", "u_r,x"),
    )
    mean_start = 50.0
    print("Time-averaged bubbly-flow quantities for t > %g:" % mean_start)
    for data, case_label in zip(datasets, case_labels):
        entries = []
        for field, terminal_label in mean_fields:
            time_average = late_time_mean(data, field, mean_start)
            if time_average is not None:
                entries.append("%s=%.6g" % (terminal_label, time_average))
        if entries:
            print("  %s: %s" % (case_label, ", ".join(entries)))
        else:
            print("  %s: no samples extend beyond t=%g" %
                  (case_label, mean_start))

    normalized_interface_thickness = []
    for data, label in zip(datasets, case_labels):
        initial_thickness = data["interfaceThickness"][0]
        if abs(initial_thickness) <= np.finfo(float).tiny:
            raise ValueError("Initial interface thickness is zero for %s" %
                             label)
        normalized_interface_thickness.append(
            data["interfaceThickness"]/initial_thickness)

    plotnow(
        os.path.join(output_dir, "gasVolumeFraction"),
        r"$t$", r"$\alpha_g$",
        times, [data["alphaG"] for data in datasets],
        case_labels, linestyles=linestyles, markers=markers,
        colors=case_colors,
    )

    positive_error = [
        np.maximum(data["relativeVolumeError"], np.finfo(float).tiny)
        for data in datasets]
    plotnow(
        os.path.join(output_dir, "relativeVolumeError"),
        r"$t$", r"$|V_g-V_{g,0}|/V_{g,0}$",
        times, positive_error,
        case_labels, linestyles=linestyles, markers=markers,
        colors=case_colors,
        ptype="semilogy",
        show_all_log_decades=True,
    )

    plot_field(
        "interfaceArea", "area", r"interface area $A$")

    plot_field(
        "sauterMeanDiameter", "d32", r"Sauter mean diameter $d_{32}$")

    plotnow(
        os.path.join(output_dir, "interfaceThickness"),
        r"$t$", r"normalized interface thickness $\ell/\ell_0$",
        times, normalized_interface_thickness,
        case_labels, linestyles=linestyles, markers=markers,
        colors=case_colors,
    )

    kinetic_energy_fields = ("kL", "kG")
    kinetic_energy_labels = (r"$k_l$", r"$k_g$")
    plotnow(
        os.path.join(output_dir, "kineticEnergy"),
        r"$t$", "kinetic energy",
        [data["t"] for field in kinetic_energy_fields for data in datasets],
        [data[field] for field in kinetic_energy_fields for data in datasets],
        ["%s, %s" % (field_label, case_label)
         for field_label in kinetic_energy_labels
         for case_label in case_labels],
        linestyles=linestyles*len(kinetic_energy_fields),
        markers=markers*len(kinetic_energy_fields),
        colors=field_colors(len(kinetic_energy_fields)),
        ptype="semilogy",
        legend_fontsize=12,
        quantity_legend_labels=kinetic_energy_labels,
        case_legend_labels=case_labels,
    )

    plotnow(
        os.path.join(output_dir, "surfaceEnergy"),
        r"$t$", "surface energy",
        times,
        [data["surfaceEnergy"] for data in datasets],
        case_labels,
        linestyles=linestyles,
        markers=markers,
        colors=case_colors,
        ptype="semilogy",
        legend_fontsize=12,
    )

    phase_fields = ("uLx", "uGx")
    phase_field_labels = (r"$\langle u_x\rangle_l$",
                          r"$\langle u_x\rangle_g$")
    phase_times = []
    phase_values = []
    phase_labels = []
    phase_linestyles = []
    phase_colors = []
    for field_index, (field, field_label) in enumerate(
            zip(phase_fields, phase_field_labels)):
        for data, case_label, linestyle in zip(
                datasets, case_labels, linestyles):
            color = palette[field_index % len(palette)]
            label = "%s, %s" % (field_label, case_label)

            phase_times.append(data["t"])
            phase_values.append(data[field])
            phase_labels.append(label)
            phase_linestyles.append(linestyle)
            phase_colors.append(color)

    plotnow(
        os.path.join(output_dir, "phaseVelocityX"),
        r"$t$", "streamwise phase velocity",
        phase_times,
        phase_values,
        phase_labels,
        linestyles=phase_linestyles,
        markers=[""]*len(phase_times),
        colors=phase_colors,
        quantity_legend_labels=phase_field_labels,
        case_legend_labels=case_labels,
    )

    plotnow(
        os.path.join(output_dir, "phaseVelocityYZ"),
        r"$t$", "transverse phase velocity",
        [data["t"] for field in ("uLy", "uLz", "uGy", "uGz")
         for data in datasets],
        [data[field] for field in ("uLy", "uLz", "uGy", "uGz")
         for data in datasets],
        ["%s, %s" % (field_label, case_label)
         for field_label in (r"$\langle u_y\rangle_l$",
                             r"$\langle u_z\rangle_l$",
                             r"$\langle u_y\rangle_g$",
                             r"$\langle u_z\rangle_g$")
         for case_label in case_labels],
        linestyles=linestyles*4, markers=markers*4,
        colors=field_colors(4),
        quantity_legend_labels=(r"$\langle u_y\rangle_l$",
                                r"$\langle u_z\rangle_l$",
                                r"$\langle u_y\rangle_g$",
                                r"$\langle u_z\rangle_g$"),
        case_legend_labels=case_labels,
    )

    plot_field(
        "slipVelocityX", "slipX",
        r"streamwise slip velocity $u_{r,x}$")

    plotnow(
        os.path.join(output_dir, "slipVelocityYZ"),
        r"$t$", "transverse slip velocity",
        [data["t"] for field in ("slipY", "slipZ")
         for data in datasets],
        [data[field] for field in ("slipY", "slipZ")
         for data in datasets],
        ["%s, %s" % (field_label, case_label)
         for field_label in (r"$u_{r,y}$", r"$u_{r,z}$")
         for case_label in case_labels],
        linestyles=linestyles*2, markers=markers*2,
        colors=field_colors(2),
        quantity_legend_labels=(r"$u_{r,y}$", r"$u_{r,z}$"),
        case_legend_labels=case_labels,
    )


def main():
    args = arguments()
    if len(args.input) != len(args.labels):
        raise ValueError("--input and --labels must have the same length")
    datasets = [read_bubbly(filename, args.t_min, args.t_max)
                for filename in args.input]
    plot_bubbly(args.output, datasets, args.labels)
    for filename, label, data in zip(args.input, args.labels, datasets):
        print("Read %d samples from %s (%s)" %
              (data["t"].size, filename, label))
    print("Wrote plots to %s" % args.output)


if __name__ == "__main__":
    starttime = time.time()
    main()
    print("--- Code ran in %s seconds ---" % (time.time() - starttime))
