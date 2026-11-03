"""Plot CHANNEL, THERMAL, y+, and runtime diagnostics from nekRS logs."""

from cycler import cycler
import argparse
import os
import re
import time

import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
KEY_VALUE = re.compile(
    r"([A-Za-z][A-Za-z0-9_]*(?:-[A-Za-z][A-Za-z0-9_]*)?)\s*=\s*(" +
    FLOAT + r")")
YPLUS = re.compile(
    r"y\+\s+min:\s*(%s)\s+max:\s*(%s)\s+avg:\s*(%s)" %
    (FLOAT, FLOAT, FLOAT))
STEP_TIME = re.compile(
    r"step=\s*(\d+)\s+t=\s*(%s)\s+dt=(%s)\s+CFL=\s*(%s)" %
    (FLOAT, FLOAT, FLOAT))
ELAPSED_STEP = re.compile(
    r"step=\s*(\d+)\s+elapsedStep=\s*(%s)s" % FLOAT)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", nargs="+", default=["hrefine/p7/log3"],
        help=("Input cases. Join chronological restart logs for one case "
              "with '+', for example 'run/log1+run/log2+run/log3'"))
    parser.add_argument("--labels", nargs="+", default=["h/2, N=7"],
                        help="Labels corresponding to the input files")
    parser.add_argument("--output", default="logPlots",
                        help="Output directory (default: logPlots)")
    parser.add_argument("--t-min", type=float, default=None,
                        help="Discard samples before this simulation time")
    parser.add_argument("--t-max", type=float, default=None,
                        help="Discard samples after this simulation time")
    return parser.parse_args()


def key_values(line):
    return {key: float(value) for key, value in KEY_VALUE.findall(line)}


def sorted_unique(records, t_min=None, t_max=None):
    """Sort by time and retain the last record at duplicate restart times."""
    by_time = {}
    for record in records:
        sample_time = record["t"]
        if t_min is not None and sample_time < t_min:
            continue
        if t_max is not None and sample_time > t_max:
            continue
        by_time[sample_time] = record
    return [by_time[sample_time] for sample_time in sorted(by_time)]


def read_log(filename, t_min=None, t_max=None):
    channel = []
    thermal = []
    runtime = []
    latest_runtime_by_step = {}
    pending_yplus = None

    with open(filename, "r", errors="replace") as logfile:
        for line in logfile:
            yplus_match = YPLUS.search(line)
            if yplus_match:
                pending_yplus = tuple(float(value)
                                      for value in yplus_match.groups())
                continue

            if line.startswith("CHANNEL"):
                record = key_values(line)
                if "t" in record:
                    if pending_yplus is not None:
                        record["yPlusMin"] = pending_yplus[0]
                        record["yPlusMax"] = pending_yplus[1]
                        record["yPlusAvg"] = pending_yplus[2]
                    channel.append(record)
                pending_yplus = None
                continue

            if line.startswith("THERMAL"):
                record = key_values(line)
                if "t" in record:
                    thermal.append(record)
                continue

            step_match = STEP_TIME.search(line)
            if step_match:
                step = int(step_match.group(1))
                record = {
                    "step": step,
                    "t": float(step_match.group(2)),
                    "dt": float(step_match.group(3)),
                    "CFL": float(step_match.group(4)),
                }
                runtime.append(record)
                latest_runtime_by_step[step] = record
                continue

            elapsed_match = ELAPSED_STEP.search(line)
            if elapsed_match:
                step = int(elapsed_match.group(1))
                if step in latest_runtime_by_step:
                    latest_runtime_by_step[step]["elapsedStep"] = float(
                        elapsed_match.group(2))

    result = {
        "channel": sorted_unique(channel, t_min, t_max),
        "thermal": sorted_unique(thermal, t_min, t_max),
        "runtime": sorted_unique(runtime, t_min, t_max),
    }
    if not any(result.values()):
        raise RuntimeError("No recognized diagnostics found in %s" % filename)
    return result


def read_log_group(input_specification, t_min=None, t_max=None):
    """Stitch '+'-separated restart logs, giving later logs precedence."""
    filenames = [name.strip() for name in input_specification.split("+")
                 if name.strip()]
    if not filenames:
        raise ValueError("Empty input specification")

    combined = {"channel": [], "thermal": [], "runtime": []}
    for filename in filenames:
        dataset = read_log(filename)
        for stream in combined:
            new_records = dataset[stream]
            if not new_records:
                continue
            restart_time = new_records[0]["t"]
            combined[stream] = [
                record for record in combined[stream]
                if record["t"] < restart_time
            ]
            combined[stream].extend(new_records)

    return {
        stream: sorted_unique(records, t_min, t_max)
        for stream, records in combined.items()
    }


def plot_fields(filename, ylabel, datasets, case_labels, stream, fields,
                field_labels=None, ptype="line"):
    palette = ["#0072B2", "#D55E00", "#009E73", "#CC0000",
               "#990099"]
    case_linestyles = ["-", ":", "--", "-."]
    if field_labels is None:
        field_labels = fields

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_prop_cycle(
        cycler(color=palette)*cycler(linestyle=["-"])*cycler(marker=[""]))

    curves = 0
    for field_index, (field, field_label) in enumerate(
            zip(fields, field_labels)):
        for case_index, (dataset, case_label) in enumerate(
                zip(datasets, case_labels)):
            records = dataset[stream]
            selected = [record for record in records if field in record]
            if not selected:
                continue
            sample_times = np.asarray([record["t"] for record in selected])
            values = np.asarray([record[field] for record in selected])
            if len(fields) == 1:
                label = case_label
            else:
                label = "%s, %s" % (field_label, case_label)
            plotter = ax.semilogy if ptype == "semilogy" else ax.plot
            plotter(sample_times, values, label=label, linewidth=2.0,
                    color=palette[field_index % len(palette)],
                    linestyle=case_linestyles[
                        case_index % len(case_linestyles)])
            curves += 1

    if curves == 0:
        plt.close(fig)
        return False

    ax.set_xlabel(r"$t$", fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid()
    ax.legend(loc="best", fontsize=10)
    fig.savefig(filename + ".pdf", bbox_inches="tight", dpi=100)
    fig.savefig(filename + ".png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    return True


def make_plots(output_dir, datasets, case_labels):
    os.makedirs(output_dir, exist_ok=True)
    specifications = [
        ("bulkVelocity", r"bulk velocity $U_b$", "channel",
         ("Ub",), None, "line"),
        ("frictionVelocity", r"friction velocity $u_\tau$", "channel",
         ("utau",), None, "line"),
        ("frictionReynolds", r"$Re_\tau$", "channel",
         ("ReTau",), None, "line"),
        ("pressureGradient", r"pressure-gradient forcing", "channel",
         ("dpdx",), None, "line"),
        ("channelTKE", r"monitored TKE $k$", "channel",
         ("kMonitor",), None, "line"),
        ("yPlus", r"$y^+$", "channel",
         ("yPlusMin", "yPlusAvg", "yPlusMax"),
         (r"$y^+_{min}$", r"$y^+_{avg}$", r"$y^+_{max}$"), "line"),
        ("temperatureMeanRms", "temperature", "thermal",
         ("Tmean", "Trms"),
         (r"$\langle T\rangle$", r"$T_{rms}$"), "line"),
        ("bulkWallTemperature", "temperature", "thermal",
         ("Tb", "Tw"), (r"$T_b$", r"$T_w$"), "line"),
        ("wallBulkDifference", r"$T_w-T_b$", "thermal",
         ("Tw-Tb",), None, "line"),
        ("stanton", "Stanton number", "thermal",
         ("St",), None, "line"),
        ("nusselt", "Nusselt number", "thermal",
         ("Nu_delta", "Nu_H"),
         (r"$Nu_\delta$", r"$Nu_H$"), "line"),
        ("timeStep", r"time step $\Delta t$", "runtime",
         ("dt",), None, "line"),
        ("cfl", "CFL", "runtime", ("CFL",), None, "line"),
        ("elapsedStep", "wall time per step [s]", "runtime",
         ("elapsedStep",), None, "semilogy"),
    ]

    written = []
    for name, ylabel, stream, fields, labels, ptype in specifications:
        destination = os.path.join(output_dir, name)
        if plot_fields(destination, ylabel, datasets, case_labels, stream,
                       fields, labels, ptype):
            written.append(name)
    return written


def main():
    args = arguments()
    if len(args.input) != len(args.labels):
        raise ValueError("--input and --labels must have the same length")

    datasets = [read_log_group(input_specification,
                               args.t_min, args.t_max)
                for input_specification in args.input]
    written = make_plots(args.output, datasets, args.labels)

    for input_specification, label, dataset in zip(
            args.input, args.labels, datasets):
        print("Read %s (%s): %d CHANNEL, %d THERMAL, %d runtime samples" %
              (input_specification, label, len(dataset["channel"]),
               len(dataset["thermal"]), len(dataset["runtime"])))
    print("Wrote %d plot groups to %s" % (len(written), args.output))


if __name__ == "__main__":
    starttime = time.time()
    main()
    print("--- Code ran in %s seconds ---" % (time.time() - starttime))
