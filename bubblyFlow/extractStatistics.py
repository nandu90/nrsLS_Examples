"""Extract channel statistics, phase anisotropy, and SP TKE budgets.

Phase statistics use intrinsic CLS-weighted averages.  A conventional TKE
budget is generated only for a constant-property single-phase channel.  With
the available checkpoint fields, a complete two-phase budget cannot be
reconstructed and is therefore deliberately not produced.
"""

import argparse
import glob
import os
import time

from mpi4py import MPI
import numpy as np

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="interpolatedData/*.npz")
    parser.add_argument("--output", default="statisticsData")
    parser.add_argument("--re", type=float, default=5600.0,
                        help="Re based on full channel height")
    parser.add_argument("--re-tau", type=float, default=280.0)
    return parser.parse_args()


def periodic_derivative(field, spacing, axis):
    n = field.shape[axis]
    wave = 2.0*np.pi*np.fft.fftfreq(n, d=spacing)
    shape = [1]*field.ndim
    shape[axis] = n
    transformed = np.fft.fft(field, axis=axis)
    return np.fft.ifft(1j*wave.reshape(shape)*transformed,
                       axis=axis).real


def strain_rate(fluctuations, y, dx, dz):
    """Return S'_ij with shape (nx,ny,nz,3,3)."""
    gradient = np.empty(fluctuations.shape + (3,))
    for component in range(3):
        value = fluctuations[..., component]
        gradient[..., component, 0] = periodic_derivative(value, dx, 0)
        gradient[..., component, 1] = np.gradient(
            value, y, axis=1, edge_order=2)
        gradient[..., component, 2] = periodic_derivative(value, dz, 2)
    return 0.5*(gradient + np.swapaxes(gradient, -1, -2))


def phase_results(weight, first, second):
    safe = np.maximum(weight, 1.0e-30)
    mean = first/safe[:, None]
    raw_second = second/safe[:, None, None]
    reynolds = raw_second - mean[:, :, None]*mean[:, None, :]
    reynolds = 0.5*(reynolds + np.swapaxes(reynolds, 1, 2))
    tke = 0.5*np.trace(reynolds, axis1=1, axis2=2)

    anisotropy = np.full_like(reynolds, np.nan)
    ii = np.full(weight.size, np.nan)
    iii = np.full(weight.size, np.nan)
    flatness = np.full(weight.size, np.nan)
    eigenvalues = np.full((weight.size, 3), np.nan)
    barycentric = np.full((weight.size, 3), np.nan)
    bary_x = np.full(weight.size, np.nan)
    bary_y = np.full(weight.size, np.nan)

    for iy in range(weight.size):
        if weight[iy] <= 1.0e-14 or tke[iy] <= 1.0e-14:
            continue
        tensor = reynolds[iy]/(2.0*tke[iy]) - np.eye(3)/3.0
        tensor = 0.5*(tensor + tensor.T)
        anisotropy[iy] = tensor
        ii[iy] = -0.5*np.sum(tensor*tensor.T)
        iii[iy] = np.linalg.det(tensor)
        flatness[iy] = 1.0 + 9.0*ii[iy] + 27.0*iii[iy]

        lam = np.linalg.eigvalsh(tensor)[::-1]
        eigenvalues[iy] = lam
        c1 = lam[0] - lam[1]
        c2 = 2.0*(lam[1] - lam[2])
        c3 = 3.0*lam[2] + 1.0
        barycentric[iy] = (c1, c2, c3)
        bary_x[iy] = c1 + 0.5*c3
        bary_y[iy] = 0.5*np.sqrt(3.0)*c3

    return {
        "mean": mean,
        "reynolds": reynolds,
        "tke": tke,
        "anisotropy": anisotropy,
        "II": ii,
        "III": iii,
        "flatness": flatness,
        "eigenvalues": eigenvalues,
        "barycentric": barycentric,
        "barycentricX": bary_x,
        "barycentricY": bary_y,
    }


def reduce_sum(comm, value):
    result = np.empty_like(value)
    comm.Allreduce(value, result, op=MPI.SUM)
    return result


def plot_results(output, y, yplus, liquid, gas, budget=None):
    upper = y >= -1.0e-12
    order = np.argsort(yplus[upper])
    yp = yplus[upper][order]

    def symmetric(profile):
        folded = 0.5*(profile + profile[::-1])
        return folded[upper][order]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for component, label in enumerate((r"$b_{11}$", r"$b_{22}$", r"$b_{33}$")):
        axes[0].semilogx(
            yp, symmetric(liquid["anisotropy"][:, component, component]),
            label=label)
    axes[0].axhline(0.0, color="black", linewidth=0.8)
    axes[0].set_xlabel(r"$y^+$")
    axes[0].set_ylabel(r"$b_{ii}$, liquid phase")
    axes[0].grid(True, which="both", alpha=0.35)
    axes[0].legend()

    axes[1].semilogx(yp, symmetric(liquid["flatness"]), label="liquid")
    if np.any(np.isfinite(gas["flatness"])):
        axes[1].semilogx(yp, symmetric(gas["flatness"]), label="gas")
    axes[1].set_xlabel(r"$y^+$")
    axes[1].set_ylabel(r"Lumley flatness $F$")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, which="both", alpha=0.35)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "anisotropyProfiles.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5.4))
    triangle_x = [0.0, 1.0, 0.5, 0.0]
    triangle_y = [0.0, 0.0, np.sqrt(3.0)/2.0, 0.0]
    ax.plot(triangle_x, triangle_y, "k-", linewidth=1.3)
    points = ax.scatter(liquid["barycentricX"], liquid["barycentricY"],
                        c=yplus, cmap="viridis", s=18)
    ax.text(1.02, 0.0, "1C", va="center")
    ax.text(-0.08, 0.0, "2C", va="center")
    ax.text(0.5, np.sqrt(3.0)/2.0 + 0.035, "3C / isotropic",
            ha="center")
    fig.colorbar(points, ax=ax, label=r"$y^+$")
    ax.set_aspect("equal")
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, 0.95)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(output, "barycentricLiquid.png"), dpi=200)
    plt.close(fig)

    if budget is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        for name, label in (("production", "production"),
                            ("dissipation", "dissipation"),
                            ("turbulentTransport", "turbulent transport"),
                            ("pressureTransport", "pressure transport"),
                            ("viscousDiffusion", "viscous diffusion"),
                            ("residual", "resolved residual")):
            ax.plot(yp, symmetric(budget[name]), label=label)
        ax.set_xlabel(r"$y^+$")
        ax.set_ylabel("TKE budget term")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(output, "tkeBudget.png"), dpi=200)
        plt.close(fig)


def main():
    args = arguments()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    files = sorted(glob.glob(args.input))
    if not files:
        raise RuntimeError("No files match %r" % args.input)

    with np.load(files[0]) as first:
        x, y, z = first["x"], first["y"], first["z"]
        shape = first["u"].shape
    nx, ny, nz = shape
    plane_points = nx*nz
    local_count = 0

    # Raw profile moments for unconditioned, liquid, and gas statistics.
    w = {name: np.zeros(ny) for name in ("all", "liquid", "gas")}
    m1 = {name: np.zeros((ny, 3)) for name in w}
    m2 = {name: np.zeros((ny, 3, 3)) for name in w}
    pressure_sum = np.zeros(ny)
    alpha_sum = np.zeros(ny)
    local_times = []

    for filename in files[rank::size]:
        with np.load(filename) as data:
            velocity = np.stack((data["u"], data["v"], data["w"]), axis=-1)
            cls = np.clip(data["cls"], 0.0, 1.0)
            weights = {
                "all": np.ones_like(cls),
                "liquid": cls,
                "gas": 1.0 - cls,
            }
            for name, weight in weights.items():
                w[name] += np.sum(weight, axis=(0, 2))
                m1[name] += np.sum(weight[..., None]*velocity, axis=(0, 2))
                m2[name] += np.einsum(
                    "xyza,xyzb,xyz->yab", velocity, velocity, weight)

            pressure = data["p"] - np.mean(data["p"])
            pressure_sum += np.sum(pressure, axis=(0, 2))
            alpha_sum += np.sum(1.0 - cls, axis=(0, 2))
            local_count += 1
            local_times.append(float(data["time"]))

    for name in w:
        w[name] = reduce_sum(comm, w[name])
        m1[name] = reduce_sum(comm, m1[name])
        m2[name] = reduce_sum(comm, m2[name])
    pressure_sum = reduce_sum(comm, pressure_sum)
    alpha_sum = reduce_sum(comm, alpha_sum)
    count = comm.allreduce(local_count, op=MPI.SUM)
    all_times = comm.gather(local_times, root=0)

    statistics = {name: phase_results(w[name], m1[name], m2[name])
                  for name in w}
    mean_velocity = statistics["all"]["mean"]
    mean_pressure = pressure_sum/(count*plane_points)
    alpha_gas = alpha_sum/(count*plane_points)
    reynolds = statistics["all"]["reynolds"]
    tke = statistics["all"]["tke"]
    two_phase = np.max(alpha_gas) > 1.0e-8
    budget = None

    if not two_phase:
        # A second pass supplies the third moments and gradients needed by the
        # conventional constant-property single-phase channel budget.
        turbulent_flux = np.zeros(ny)
        pressure_flux = np.zeros(ny)
        dissipation = np.zeros(ny)
        dx = float(x[1] - x[0])
        dz = float(z[1] - z[0])
        viscosity = 2.0/args.re

        for filename in files[rank::size]:
            with np.load(filename) as data:
                velocity = np.stack(
                    (data["u"], data["v"], data["w"]), axis=-1)
                fluctuation = velocity - mean_velocity[None, :, None, :]
                q2 = np.sum(fluctuation**2, axis=-1)
                turbulent_flux += np.sum(
                    0.5*fluctuation[..., 1]*q2, axis=(0, 2))

                pressure = data["p"] - np.mean(data["p"])
                pressure_prime = pressure - mean_pressure[None, :, None]
                pressure_flux += np.sum(
                    pressure_prime*fluctuation[..., 1], axis=(0, 2))

                strain = strain_rate(fluctuation, y, dx, dz)
                instantaneous_dissipation = 2.0*viscosity*np.sum(
                    strain*strain, axis=(-1, -2))
                dissipation += np.sum(
                    instantaneous_dissipation, axis=(0, 2))

        turbulent_flux = reduce_sum(
            comm, turbulent_flux)/(count*plane_points)
        pressure_flux = reduce_sum(
            comm, pressure_flux)/(count*plane_points)
        dissipation = reduce_sum(
            comm, dissipation)/(count*plane_points)

        mean_gradient = np.gradient(
            mean_velocity, y, axis=0, edge_order=2)
        production = -np.sum(
            reynolds[:, :, 1]*mean_gradient, axis=1)
        turbulent_transport = -np.gradient(
            turbulent_flux, y, edge_order=2)
        pressure_transport = -np.gradient(
            pressure_flux, y, edge_order=2)
        viscous_diffusion = viscosity*np.gradient(
            np.gradient(tke, y, edge_order=2), y, edge_order=2)
        residual = (production - dissipation + turbulent_transport
                    + pressure_transport + viscous_diffusion)
        budget = {
            "production": production,
            "dissipation": -dissipation,
            "turbulentTransport": turbulent_transport,
            "pressureTransport": pressure_transport,
            "viscousDiffusion": viscous_diffusion,
            "residual": residual,
        }

    if rank != 0:
        return

    os.makedirs(args.output, exist_ok=True)
    times = np.sort(np.concatenate([np.asarray(t) for t in all_times]))
    yplus = args.re_tau*(1.0 - np.abs(y))
    output = {
        "time": times, "y": y, "yplus": yplus,
        "alphaGas": alpha_gas,
        "meanVelocity": mean_velocity,
        "meanPressure": mean_pressure,
        "reynoldsStress": reynolds,
        "tke": tke,
        "reTau": np.array(args.re_tau),
        "twoPhase": np.array(two_phase),
    }
    for phase in ("liquid", "gas"):
        for name, value in statistics[phase].items():
            output[phase + name[0].upper() + name[1:]] = value
    if budget is not None:
        for name, value in budget.items():
            output["budget" + name[0].upper() + name[1:]] = value
    np.savez(os.path.join(args.output, "statistics.npz"), **output)

    profile = np.column_stack((
        y, yplus, alpha_gas,
        statistics["liquid"]["tke"],
        statistics["liquid"]["anisotropy"][:, 0, 0],
        statistics["liquid"]["anisotropy"][:, 1, 1],
        statistics["liquid"]["anisotropy"][:, 2, 2],
        statistics["liquid"]["flatness"],
        statistics["liquid"]["barycentricX"],
        statistics["liquid"]["barycentricY"],
    ))
    np.savetxt(os.path.join(args.output, "anisotropyProfile.dat"), profile,
               header="y yplus alphaGas kLiquid b11 b22 b33 F xB yB")

    if budget is not None:
        budget_table = np.column_stack((
            y, yplus, production, -dissipation, turbulent_transport,
            pressure_transport, viscous_diffusion, residual))
        np.savetxt(os.path.join(args.output, "tkeBudget.dat"), budget_table,
                   header=("y yplus production minusDissipation "
                           "turbulentTransport pressureTransport "
                           "viscousDiffusion resolvedResidual"))
    plot_results(args.output, y, yplus, statistics["liquid"],
                 statistics["gas"], budget)
    print("Wrote statistics to %s" % args.output)
    if two_phase:
        print("Two-phase case detected: no TKE budget was generated because "
              "the available checkpoint fields cannot close it.")


if __name__ == "__main__":
    start = time.time()
    main()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("--- statistics completed in %.2f s ---" %
              (time.time() - start))
