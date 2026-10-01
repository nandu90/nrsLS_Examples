"""Combine compact in-situ budgetProfiles files and plot channel profiles."""

import argparse
import glob
import os
import re

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


FIELDS = (
    "u", "v", "w", "rho", "mu", "cls", "fSigmaX", "fSigmaY",
    "fSigmaZ", "surfaceTensionPower", "viscousDissipation",
    "pressureGradientPower", "gravityPower", "tau11", "tau12", "tau13",
    "tau22", "tau23", "tau33", "uFSigmaX", "vFSigmaY", "wFSigmaZ",
    "rhoU", "rhoV", "rhoW",
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="budgetProfiles*.dat")
    parser.add_argument("--output", default="inSituBudgetData")
    parser.add_argument("--re-tau", type=float, default=280.0)
    parser.add_argument("--coordinate-decimals", type=int, default=12)
    return parser.parse_args()


def read_blocks(filename):
    names, block, rows = None, None, []
    with open(filename, encoding="utf-8") as stream:
        for raw_line in stream:
            line = raw_line.strip()
            if not line:
                if block is not None and rows:
                    yield names, block, np.asarray(rows, dtype=float)
                    block, rows = None, []
                continue
            if line.startswith("# y "):
                names = tuple(line[2:].split()[1:])
            elif line.startswith("# intervalEnd"):
                if block is not None and rows:
                    yield names, block, np.asarray(rows, dtype=float)
                    rows = []
                match = re.fullmatch(
                    r"# intervalEnd (\S+) duration (\S+) points (\d+)", line)
                if not match:
                    raise ValueError("Malformed interval header in %s" % filename)
                block = {"end": float(match.group(1)),
                         "duration": float(match.group(2)),
                         "points": int(match.group(3))}
            elif not line.startswith("#"):
                if block is None:
                    raise ValueError("Data before interval header in %s" % filename)
                rows.append([float(value) for value in line.split()])
    if block is not None and rows:
        yield names, block, np.asarray(rows, dtype=float)


def collapse_duplicate_y(data, decimals):
    keys = np.round(data[:, 0], decimals)
    y, inverse = np.unique(keys, return_inverse=True)
    count = np.bincount(inverse)
    values = np.empty((y.size, data.shape[1] - 1))
    for column in range(values.shape[1]):
        values[:, column] = np.bincount(
            inverse, weights=data[:, column + 1])/count
    return y, values


def write_outputs(output, y, profiles, duration, re_tau):
    os.makedirs(output, exist_ok=True)
    profiles["alphaGas"] = 1.0 - profiles["cls"]
    profiles["surfaceTensionPowerMean"] = (
        profiles["u"]*profiles["fSigmaX"]
        + profiles["v"]*profiles["fSigmaY"]
        + profiles["w"]*profiles["fSigmaZ"])
    profiles["surfaceTensionPowerTurbulent"] = (
        profiles["uFSigmaX"] + profiles["vFSigmaY"]
        + profiles["wFSigmaZ"] - profiles["surfaceTensionPowerMean"])
    for component in "UVW":
        profiles["favre" + component] = (
            profiles["rho" + component]/profiles["rho"])

    np.savez_compressed(
        os.path.join(output, "inSituBudgetProfiles.npz"),
        y=y, yPlus=(1.0 - np.abs(y))*re_tau,
        averagingDuration=np.array(duration), **profiles)

    columns = (
        "y", "alphaGas", "rho", "u", "v", "w", "favreU",
        "fSigmaX", "fSigmaY", "fSigmaZ", "surfaceTensionPower",
        "surfaceTensionPowerMean", "surfaceTensionPowerTurbulent",
        "viscousDissipation", "pressureGradientPower", "gravityPower",
        "tau11", "tau12", "tau13", "tau22", "tau23", "tau33",
    )
    table = np.column_stack([y] + [profiles[name] for name in columns[1:]])
    np.savetxt(os.path.join(output, "inSituBudgetProfiles.dat"), table,
               header=" ".join(columns))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(profiles["alphaGas"], y)
    axes[0].set_xlabel(r"$\langle\alpha_g\rangle_{xz,t}$")
    axes[0].set_ylabel(r"$y/\delta$")
    for name, label in (
            ("surfaceTensionPowerTurbulent", "capillary, fluctuating"),
            ("surfaceTensionPower", "capillary, total"),
            ("viscousDissipation", "viscous dissipation"),
            ("pressureGradientPower", "pressure-gradient power"),
            ("gravityPower", "gravity power")):
        axes[1].plot(profiles[name], y, label=label)
    axes[1].set_xlabel("plane- and time-averaged power density")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(True, alpha=0.35)
    fig.tight_layout()
    fig.savefig(os.path.join(output, "inSituBudgetProfiles.png"), dpi=200)
    plt.close(fig)


def main():
    args = arguments()
    files = sorted(glob.glob(args.input))
    if not files:
        raise RuntimeError("No files match %r" % args.input)
    accumulated = reference_y = None
    total_duration, intervals = 0.0, 0
    for filename in files:
        for names, metadata, raw_data in read_blocks(filename):
            if names != FIELDS:
                raise RuntimeError("Unexpected field ordering in %s" % filename)
            if raw_data.shape != (metadata["points"], len(names) + 1):
                raise RuntimeError("Incomplete interval in %s at time %.8g"
                                   % (filename, metadata["end"]))
            y, values = collapse_duplicate_y(
                raw_data, args.coordinate_decimals)
            if reference_y is None:
                reference_y, accumulated = y, np.zeros_like(values)
            if y.shape != reference_y.shape or not np.allclose(y, reference_y):
                raise RuntimeError("Incompatible y grids in %s" % filename)
            accumulated += metadata["duration"]*values
            total_duration += metadata["duration"]
            intervals += 1
    if total_duration <= 0.0:
        raise RuntimeError("No positive-duration profile intervals found")
    profiles = {name: accumulated[:, i]/total_duration
                for i, name in enumerate(FIELDS)}
    write_outputs(args.output, reference_y, profiles,
                  total_duration, args.re_tau)
    print("Combined %d intervals over duration %.8g into %s"
          % (intervals, total_duration, args.output))


if __name__ == "__main__":
    main()
