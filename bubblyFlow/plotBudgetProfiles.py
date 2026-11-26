"""Read and plot the interval-averaged profiles in budgetProfiles.dat."""

from cycler import cycler
import argparse
import glob
import os
import re
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
        help=("Input profile file, directory containing budgetProfiles*.dat, "
              "or '+'-separated chronological restart files/directories. "
              "Supply one item per case to compare multiple cases"))
    parser.add_argument("--labels", nargs="+", default=["case"],
                        help="Labels corresponding to the input cases")
    parser.add_argument("--output", default="budgetProfilePlots")
    parser.add_argument("--t-min", type=float, default=None,
                        help="Use intervals ending at or after this time")
    parser.add_argument("--t-max", type=float, default=None,
                        help="Use intervals ending at or before this time")
    parser.add_argument("--last", action="store_true",
                        help="Plot only the last selected interval")
    parser.add_argument(
        "--single-phase-statistics", default=None,
        help=("Optional statistics.npz file, or its containing directory, "
              "whose single-phase mean velocity and temperature profiles "
              "are overlaid on the corresponding figures"))
    parser.add_argument("--single-phase-label", default="single phase",
                        help="Legend label for --single-phase-statistics")
    parser.add_argument(
        "--re-tau", nargs="+", type=float, default=None,
        help=("Friction Reynolds number for each budget-profile case. "
              "A single value is applied to every case; default: 280"))
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


def natural_sort_key(path):
    """Sort embedded integers numerically, e.g. file9 before file10."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path)]


def expand_profile_input(input_specification):
    """Expand directory terms to naturally ordered profile restart files."""
    filenames = []
    terms = [name.strip() for name in input_specification.split("+")
             if name.strip()]
    if not terms:
        raise ValueError("Empty input specification")
    for term in terms:
        if os.path.isdir(term):
            matches = sorted(
                (name for name in glob.glob(
                    os.path.join(term, "budgetProfiles*.dat"))
                 if os.path.isfile(name)),
                key=natural_sort_key)
            if not matches:
                raise RuntimeError(
                    "No budgetProfiles*.dat files found in %s" % term)
            filenames.extend(matches)
        else:
            filenames.append(term)
    return filenames


def read_profile_group(input_specification):
    """Stitch restart files and align legacy/current fields by name."""
    filenames = expand_profile_input(input_specification)

    reference_names = []
    combined = []
    for filename in filenames:
        names, blocks = read_profiles(filename)

        union_names = reference_names + [name for name in names
                                         if name not in reference_names]
        if union_names != reference_names:
            old_indices = {name: index for index, name in
                           enumerate(reference_names)}
            for block in combined:
                aligned = np.full((block["points"], len(union_names)),
                                  np.nan)
                for index, name in enumerate(union_names):
                    if name in old_indices:
                        aligned[:, index] = block["values"][:,
                                                           old_indices[name]]
                block["values"] = aligned
            reference_names = union_names

        new_indices = {name: index for index, name in enumerate(names)}
        for block in blocks:
            aligned = np.full((block["points"], len(reference_names)), np.nan)
            for index, name in enumerate(reference_names):
                if name in new_indices:
                    aligned[:, index] = block["values"][:, new_indices[name]]
            block["values"] = aligned

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
    field_duration = np.zeros_like(blocks[0]["values"])
    total_duration = 0.0

    for block in blocks:
        values = block["values"]
        if values.shape != weighted_sum.shape or not np.allclose(
                values[:, 0], reference_y, rtol=1.0e-10, atol=1.0e-12):
            raise ValueError("Profile grids differ between output intervals")
        available = np.isfinite(values)
        weighted_sum += block["duration"]*np.where(available, values, 0.0)
        field_duration += block["duration"]*available
        total_duration += block["duration"]

    if total_duration <= 0.0:
        raise ValueError("Selected intervals have zero averaging duration")
    averaged = np.full_like(weighted_sum, np.nan)
    np.divide(weighted_sum, field_duration, out=averaged,
              where=field_duration > 0.0)
    averaged[:, 0] = reference_y
    profile = {name: averaged[:, index]
               for index, name in enumerate(names)
               if np.any(field_duration[:, index] > 0.0)}
    durations = {name: float(np.max(field_duration[:, index]))
                 for index, name in enumerate(names)
                 if np.any(field_duration[:, index] > 0.0)}
    return profile, total_duration, durations


def load_single_phase_profile(path):
    """Load mean velocity and temperature from extractStatistics output."""
    if os.path.isdir(path):
        path = os.path.join(path, "statistics.npz")
    with np.load(path) as statistics:
        required = ("y", "yplus", "meanVelocity", "meanTemperature")
        missing = [name for name in required if name not in statistics.files]
        if missing:
            raise RuntimeError(
                "%s is missing required field(s): %s" %
                (path, ", ".join(missing)))
        mean_velocity = statistics["meanVelocity"]
        if mean_velocity.ndim != 2 or mean_velocity.shape[1] < 1:
            raise ValueError("meanVelocity in %s has unexpected shape %s" %
                             (path, mean_velocity.shape))
        profile = {
            "y": statistics["y"].copy(),
            "yplus": statistics["yplus"].copy(),
            "u": mean_velocity[:, 0].copy(),
            "T": statistics["meanTemperature"].copy(),
        }
    if not (profile["y"].shape == profile["yplus"].shape ==
            profile["u"].shape == profile["T"].shape):
        raise ValueError("Single-phase profile arrays in %s do not align" %
                         path)
    return profile, path


def plot_profiles(output_dir, profiles, case_labels,
                  single_phase_profile=None,
                  single_phase_label="single phase", case_re_tau=None):
    os.makedirs(output_dir, exist_ok=True)
    if case_re_tau is None:
        case_re_tau = [280.0]*len(profiles)
    case_styles = ["-", ":", "--", "-."]
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC0000",
               "#990099", "#56B4E9"]

    def comparison(name, ylabel, fields, field_labels, transforms=None,
                   include_single_phase=False):
        x_values, y_values, labels = [], [], []
        linestyles, colors = [], []
        if transforms is None:
            transforms = [lambda profile, field=field: profile[field]
                          for field in fields]
        plotted_profiles = list(profiles)
        plotted_labels = list(case_labels)
        if include_single_phase and single_phase_profile is not None:
            plotted_profiles.append(single_phase_profile)
            plotted_labels.append(single_phase_label)
        for field_index, (field_label, transform) in enumerate(
                zip(field_labels, transforms)):
            for case_index, (profile, case_label) in enumerate(
                    zip(plotted_profiles, plotted_labels)):
                x_values.append(profile["y"])
                y_values.append(transform(profile))
                labels.append(case_label if len(fields) == 1 else
                              "%s, %s" % (field_label, case_label))
                linestyles.append(
                    case_styles[case_index % len(case_styles)])
                if len(fields) == 1:
                    colors.append(palette[case_index % len(palette)])
                else:
                    colors.append(palette[field_index % len(palette)])
        plotnow(
            os.path.join(output_dir, name), r"$y/\delta$", ylabel,
            x_values, y_values, labels,
            linestyles=linestyles, markers=[""]*len(y_values),
            colors=colors,
        )

    def folded_wall_curve(profile, field, re_tau=None):
        """Fold an even profile about the centerline and return y+ > 0."""
        y = np.asarray(profile["y"])
        values = np.asarray(profile[field])
        order = np.argsort(y)
        y = y[order]
        values = values[order]
        upper = y >= -1.0e-12
        upper_y = y[upper]
        upper_values = values[upper]
        lower_values = np.interp(-upper_y, y, values)
        folded_values = 0.5*(upper_values + lower_values)

        if "yplus" in profile:
            stored_yplus = np.asarray(profile["yplus"])[order]
            wall_coordinate = stored_yplus[upper]
        else:
            wall_coordinate = re_tau*(1.0 - np.abs(upper_y))

        valid = ((wall_coordinate > 0.0) &
                 np.isfinite(wall_coordinate) &
                 np.isfinite(folded_values))
        wall_coordinate = wall_coordinate[valid]
        folded_values = folded_values[valid]
        wall_order = np.argsort(wall_coordinate)
        return wall_coordinate[wall_order], folded_values[wall_order]

    def wall_comparison(name, ylabel, field, include_single_phase=True,
                        transform=None, ptype="semilogx"):
        wall_profiles = list(profiles)
        wall_labels = list(case_labels)
        wall_re_tau = list(case_re_tau)
        if include_single_phase and single_phase_profile is not None:
            wall_profiles.append(single_phase_profile)
            wall_labels.append(single_phase_label)
            wall_re_tau.append(None)

        x_values, y_values = [], []
        for profile, re_tau in zip(wall_profiles, wall_re_tau):
            wall_coordinate, folded_values = folded_wall_curve(
                profile, field, re_tau)
            if transform is not None:
                folded_values = transform(folded_values)
            x_values.append(wall_coordinate)
            y_values.append(folded_values)
        plotnow(
            os.path.join(output_dir, name), r"$y^+$", ylabel,
            x_values, y_values, wall_labels, ptype=ptype,
            linestyles=[case_styles[index % len(case_styles)]
                        for index in range(len(wall_profiles))],
            markers=[""]*len(wall_profiles),
            colors=[palette[index % len(palette)]
                    for index in range(len(wall_profiles))],
        )

    # Requested single-quantity mesh comparisons.
    comparison("density", r"$\langle\rho\rangle_{xz,t}$",
               ("rho",), (r"$\langle\rho\rangle$",))
    comparison("phaseIndicator", r"$\langle\psi\rangle_{xz,t}$",
               ("cls",), (r"$\langle\psi\rangle$",))
    wall_comparison("phaseIndicatorYplus", r"$\langle\psi\rangle_{xz,t}$",
                    "cls", include_single_phase=False)
    # CLS equals one in the continuous liquid and zero in the dispersed gas,
    # so its complement is the local gas volume (void) fraction.  Retain the
    # raw phase-indicator plots above and additionally provide the quantity
    # conventionally reported by bubbly-channel studies.
    comparison(
        "voidFraction", r"$\langle\alpha_g\rangle_{xz,t}$",
        ("cls",), (r"$\langle\alpha_g\rangle$",),
        transforms=(lambda profile: 1.0 - profile["cls"],))
    wall_comparison(
        "voidFractionYplus", r"$\langle\alpha_g\rangle_{xz,t}$",
        "cls", include_single_phase=False,
        transform=lambda phase_indicator: 1.0 - phase_indicator,
        ptype="line")
    comparison("meanVelocity", r"$\langle u\rangle_{xz,t}$",
               ("u",), (r"$\langle u\rangle$",),
               include_single_phase=True)
    wall_comparison("meanVelocityYplus", r"$\langle u\rangle_{xz,t}$",
                    "u")
    comparison("densityFlux", r"$\langle\rho u\rangle_{xz,t}$",
               ("rhoU",), (r"$\langle\rho u\rangle$",))

    if all("T" in profile for profile in profiles):
        comparison("meanTemperature", r"$\langle \Theta\rangle_{xz,t}$",
                   ("T",), (r"$\langle \Theta\rangle$",),
                   include_single_phase=True)
        wall_comparison(
            "meanTemperatureYplus", r"$\langle \Theta\rangle_{xz,t}$",
            "T")
        comparison(
            "temperatureRms", r"$\Theta_{rms}$", ("temperatureRms",),
            (r"$\Theta_{rms}$",),
            transforms=(lambda p: np.sqrt(
                np.maximum(p["T2"] - p["T"]*p["T"], 0.0)),))
        comparison("conductivity", r"$\langle k\rangle_{xz,t}$",
                   ("conductivity",), (r"$\langle k\rangle$",))
        comparison("volumetricHeatCapacity",
                   r"$\langle\rho c_p\rangle_{xz,t}$",
                   ("rhoCp",), (r"$\langle\rho c_p\rangle$",))
        comparison(
            "turbulentHeatFlux", r"velocity--$\Theta$ covariance",
            ("uPrimeTPrime", "vPrimeTPrime", "wPrimeTPrime"),
            (r"$\langle u'\Theta'\rangle$",
             r"$\langle v'\Theta'\rangle$",
             r"$\langle w'\Theta'\rangle$"),
            transforms=(lambda p: p["uT"] - p["u"]*p["T"],
                        lambda p: p["vT"] - p["v"]*p["T"],
                        lambda p: p["wT"] - p["w"]*p["T"]))
        comparison(
            "convectiveEnthalpyFlux", "convective enthalpy flux",
            ("rhoCpUT", "rhoCpVT", "rhoCpWT"),
            (r"$\langle\rho c_p u\Theta\rangle$",
             r"$\langle\rho c_p v\Theta\rangle$",
             r"$\langle\rho c_p w\Theta\rangle$"))

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
    if args.re_tau is None:
        case_re_tau = [280.0]*len(args.input)
    elif len(args.re_tau) == 1:
        case_re_tau = args.re_tau*len(args.input)
    elif len(args.re_tau) == len(args.input):
        case_re_tau = args.re_tau
    else:
        raise ValueError("--re-tau requires one value or one per input case")

    profiles = []
    metadata = []
    profile_names = []
    for input_specification, label in zip(args.input, args.labels):
        names, blocks = read_profile_group(input_specification)
        selected = select_profiles(
            blocks, args.t_min, args.t_max, args.last)
        profile, duration, field_durations = average_profiles(names, selected)
        profiles.append(profile)
        profile_names.append([name for name in names if name in profile])
        metadata.append((input_specification, label, len(blocks),
                         len(selected), duration, field_durations))

    single_phase_profile = None
    single_phase_path = None
    if args.single_phase_statistics is not None:
        single_phase_profile, single_phase_path = load_single_phase_profile(
            args.single_phase_statistics)

    plot_profiles(args.output, profiles, args.labels,
                  single_phase_profile, args.single_phase_label,
                  case_re_tau)
    if single_phase_path is not None:
        print("Overlaid single-phase mean profiles from %s (%s)" %
              (single_phase_path, args.single_phase_label))
    for case_index, (profile, details) in enumerate(
            zip(profiles, metadata)):
        (input_specification, label, block_count, selected_count, duration,
         field_durations) = details
        suffix = "" if len(profiles) == 1 else "_%d" % case_index
        names = profile_names[case_index]
        output_values = np.column_stack(
            [profile[name] for name in names])
        np.savetxt(
            os.path.join(args.output, "averagedProfile%s.dat" % suffix),
            output_values, header=" ".join(names))
        print("Read %d intervals from %s (%s)" %
              (block_count, input_specification, label))
        print("Averaged %d selected intervals over duration %g" %
              (selected_count, duration))
        shorter = [(name, value) for name, value in field_durations.items()
                   if value < duration - 1.0e-12*max(1.0, duration)]
        if shorter:
            grouped = {}
            for name, value in shorter:
                grouped.setdefault(value, []).append(name)
            for value, names_at_duration in sorted(grouped.items()):
                print("  duration %g for fields: %s" %
                      (value, ", ".join(names_at_duration)))
    print("Wrote comparison plots and averaged profile data to %s" %
          args.output)


if __name__ == "__main__":
    starttime = time.time()
    main()
    print("--- Code ran in %s seconds ---" % (time.time() - starttime))
