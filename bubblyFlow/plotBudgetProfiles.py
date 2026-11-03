"""Read and plot the interval-averaged profiles in budgetProfiles.dat."""

from cycler import cycler
import argparse
import os
import time

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def plotnow(fname, xlabel, ylabel, x, y, labels, ptype="line",
            linestyles=[], markers=[], colors=[], ylim=[], xlim=[]):
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

    ax.grid()
    ax.legend(loc="best", fontsize=12)
    fig.savefig(fname + ".pdf", bbox_inches="tight", dpi=100)
    fig.savefig(fname + ".png", bbox_inches="tight", dpi=100)
    plt.close()

    return


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", default=["budgetProfiles.dat"],
        help=("Input profile file, or '+'-separated chronological restart "
              "files. Supply one item per case to compare multiple cases"))
    parser.add_argument("--labels", nargs="+", default=["case"],
                        help="Labels corresponding to the input cases")
    parser.add_argument("--output", default="budgetProfilePlots")
    parser.add_argument("--t-min", type=float, default=None,
                        help="Use intervals ending at or after this time")
    parser.add_argument("--t-max", type=float, default=None,
                        help="Use intervals ending at or before this time")
    parser.add_argument("--last", action="store_true",
                        help="Plot only the last selected interval")
    return parser.parse_args()


def read_profiles(filename):
    names = None
    blocks = []
    current = None

    with open(filename, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("# y "):
                names = stripped[2:].split()
                continue
            if stripped.startswith("# intervalEnd "):
                words = stripped[2:].split()
                if len(words) != 6 or words[0] != "intervalEnd" \
                        or words[2] != "duration" or words[4] != "points":
                    raise ValueError("Malformed interval header at line %d" %
                                     line_number)
                current = {
                    "time": float(words[1]),
                    "duration": float(words[3]),
                    "points": int(words[5]),
                    "rows": [],
                }
                blocks.append(current)
                continue
            if stripped.startswith("#"):
                continue
            if current is None:
                raise ValueError("Profile row precedes interval header at line %d" %
                                 line_number)
            current["rows"].append([float(value)
                                    for value in stripped.split()])

    if names is None:
        raise RuntimeError("No '# y ...' field header found in %s" % filename)
    if not blocks:
        raise RuntimeError("No profile intervals found in %s" % filename)

    for block in blocks:
        block["values"] = np.asarray(block.pop("rows"), dtype=float)
        expected = (block["points"], len(names))
        if block["values"].shape != expected:
            raise ValueError(
                "Interval ending at %g has shape %s; expected %s" %
                (block["time"], block["values"].shape, expected))
    return names, blocks


def read_profile_group(input_specification):
    """Stitch '+'-separated files, giving later restart files precedence."""
    filenames = [name.strip() for name in input_specification.split("+")
                 if name.strip()]
    if not filenames:
        raise ValueError("Empty input specification")

    reference_names = None
    combined = []
    for filename in filenames:
        names, blocks = read_profiles(filename)
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError("Profile fields differ in %s" % filename)

        # A block represents the half-open averaging interval
        # (intervalEnd-duration, intervalEnd].  A later restart file replaces
        # every earlier block that overlaps its first averaging interval.
        restart_time = blocks[0]["time"] - blocks[0]["duration"]
        tolerance = 1.0e-12*max(1.0, abs(restart_time))
        combined = [block for block in combined
                    if block["time"] <= restart_time + tolerance]
        combined.extend(blocks)

    # Remove exact duplicate interval-end times, retaining the later file.
    by_time = {block["time"]: block for block in combined}
    return reference_names, [by_time[key] for key in sorted(by_time)]


def select_profiles(blocks, t_min=None, t_max=None, last=False):
    selected = blocks
    if t_min is not None:
        selected = [block for block in selected if block["time"] >= t_min]
    if t_max is not None:
        selected = [block for block in selected if block["time"] <= t_max]
    if not selected:
        raise RuntimeError("No profile intervals remain after filtering")
    return selected[-1:] if last else selected


def average_profiles(names, blocks):
    reference_y = blocks[0]["values"][:, 0]
    weighted_sum = np.zeros_like(blocks[0]["values"])
    total_duration = 0.0

    for block in blocks:
        values = block["values"]
        if values.shape != weighted_sum.shape or not np.allclose(
                values[:, 0], reference_y, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("Profile grids differ between output intervals")
        weighted_sum += block["duration"]*values
        total_duration += block["duration"]

    if total_duration <= 0.0:
        raise ValueError("Selected intervals have zero averaging duration")
    averaged = weighted_sum/total_duration
    averaged[:, 0] = reference_y
    return ({name: averaged[:, index]
             for index, name in enumerate(names)}, total_duration)


def plot_profiles(output_dir, profiles, case_labels):
    os.makedirs(output_dir, exist_ok=True)
    case_styles = ["-", ":", "--", "-."]
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC0000",
               "#990099", "#56B4E9"]

    def comparison(name, ylabel, fields, field_labels, transforms=None):
        x_values, y_values, labels = [], [], []
        linestyles, colors = [], []
        if transforms is None:
            transforms = [lambda profile, field=field: profile[field]
                          for field in fields]
        for field_index, (field_label, transform) in enumerate(
                zip(field_labels, transforms)):
            for case_index, (profile, case_label) in enumerate(
                    zip(profiles, case_labels)):
                x_values.append(profile["y"])
                y_values.append(transform(profile))
                labels.append(case_label if len(fields) == 1 else
                              "%s, %s" % (field_label, case_label))
                linestyles.append(
                    case_styles[case_index % len(case_styles)])
                colors.append(palette[field_index % len(palette)])
        plotnow(
            os.path.join(output_dir, name), r"$y/\delta$", ylabel,
            x_values, y_values, labels,
            linestyles=linestyles, markers=[""]*len(y_values),
            colors=colors,
        )

    # Requested single-quantity mesh comparisons.
    comparison("density", r"$\langle\rho\rangle_{xz,t}$",
               ("rho",), (r"$\langle\rho\rangle$",))
    comparison("phaseIndicator", r"$\langle CLS\rangle_{xz,t}$",
               ("cls",), (r"$\langle CLS\rangle$",))
    comparison("meanVelocity", r"$\langle u\rangle_{xz,t}$",
               ("u",), (r"$\langle u\rangle$",))
    comparison("densityFlux", r"$\langle\rho u\rangle_{xz,t}$",
               ("rhoU",), (r"$\langle\rho u\rangle$",))

    if all("T" in profile for profile in profiles):
        comparison("meanTemperature", r"$\langle T\rangle_{xz,t}$",
                   ("T",), (r"$\langle T\rangle$",))
        comparison(
            "temperatureRms", r"$T_{rms}$", ("temperatureRms",),
            (r"$T_{rms}$",),
            transforms=(lambda p: np.sqrt(
                np.maximum(p["T2"] - p["T"]*p["T"], 0.0)),))
        comparison("conductivity", r"$\langle k\rangle_{xz,t}$",
                   ("conductivity",), (r"$\langle k\rangle$",))
        comparison("volumetricHeatCapacity",
                   r"$\langle\rho c_p\rangle_{xz,t}$",
                   ("rhoCp",), (r"$\langle\rho c_p\rangle$",))
        comparison(
            "turbulentHeatFlux", "velocity--temperature covariance",
            ("uPrimeTPrime", "vPrimeTPrime", "wPrimeTPrime"),
            (r"$\langle u'T'\rangle$", r"$\langle v'T'\rangle$",
             r"$\langle w'T'\rangle$"),
            transforms=(lambda p: p["uT"] - p["u"]*p["T"],
                        lambda p: p["vT"] - p["v"]*p["T"],
                        lambda p: p["wT"] - p["w"]*p["T"]))
        comparison(
            "convectiveEnthalpyFlux", "convective enthalpy flux",
            ("rhoCpUT", "rhoCpVT", "rhoCpWT"),
            (r"$\langle\rho c_p uT\rangle$",
             r"$\langle\rho c_p vT\rangle$",
             r"$\langle\rho c_p wT\rangle$"))

    comparison(
        "mechanicalPower", "planar-mean power density",
        ("pressureGradientPower", "gravityPower", "surfaceTensionPower",
         "viscousSink", "net"),
        ("pressure-gradient power", "gravity power",
         "surface-tension power", "viscous sink", "net"),
        transforms=(
            lambda p: p["pressureGradientPower"],
            lambda p: p["gravityPower"],
            lambda p: p["surfaceTensionPower"],
            lambda p: -p["viscousDissipation"],
            lambda p: (p["pressureGradientPower"] + p["gravityPower"] +
                       p["surfaceTensionPower"] - p["viscousDissipation"]),
        ))
    stress_fields = ("tau11", "tau12", "tau13",
                     "tau22", "tau23", "tau33")
    comparison(
        "viscousStress", "planar-mean viscous stress", stress_fields,
        tuple(r"$\tau_{%s}$" % name[3:] for name in stress_fields))


def main():
    args = arguments()
    if len(args.input) != len(args.labels):
        raise ValueError("--input and --labels must have the same length")

    profiles = []
    metadata = []
    reference_names = None
    for input_specification, label in zip(args.input, args.labels):
        names, blocks = read_profile_group(input_specification)
        if reference_names is None:
            reference_names = names
        elif names != reference_names:
            raise ValueError("Profile fields differ between cases")
        selected = select_profiles(
            blocks, args.t_min, args.t_max, args.last)
        profile, duration = average_profiles(names, selected)
        profiles.append(profile)
        metadata.append((input_specification, label, len(blocks),
                         len(selected), duration))

    plot_profiles(args.output, profiles, args.labels)
    for case_index, (profile, details) in enumerate(
            zip(profiles, metadata)):
        input_specification, label, block_count, selected_count, duration = \
            details
        suffix = "" if len(profiles) == 1 else "_%d" % case_index
        output_values = np.column_stack(
            [profile[name] for name in reference_names])
        np.savetxt(
            os.path.join(args.output, "averagedProfile%s.dat" % suffix),
            output_values, header=" ".join(reference_names))
        print("Read %d intervals from %s (%s)" %
              (block_count, input_specification, label))
        print("Averaged %d selected intervals over duration %g" %
              (selected_count, duration))
    print("Wrote comparison plots and averaged profile data to %s" %
          args.output)


if __name__ == "__main__":
    starttime = time.time()
    main()
    print("--- Code ran in %s seconds ---" % (time.time() - starttime))
