"""Interpolate NekRS channel checkpoints onto a structured analysis grid.

The resulting arrays have shape (nx, ny, nz).  The x and z coordinates are
uniform and exclude their repeated periodic endpoints; y includes both walls
and uses the same tanh mapping as channel.udf by default.
"""

import argparse
import glob
import os
import time

from mpi4py import MPI
import numpy as np

from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.interpolation.probes import Probes


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pattern", default="*0.f00*",
                        help="checkpoint glob (default: %(default)s)")
    parser.add_argument("--output", default="interpolatedData",
                        help="output directory (default: %(default)s)")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=129)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--beta", type=float, default=1.5,
                        help="wall-normal tanh stretching; zero gives uniform y")
    parser.add_argument("--max-points", type=int, default=256,
                        help="points processed per interpolation batch")
    return parser.parse_args()


def channel_grid(nx, ny, nz, beta):
    x = np.linspace(-np.pi, np.pi, nx, endpoint=False)
    eta = np.linspace(-1.0, 1.0, ny)
    if abs(beta) > 1.0e-14:
        y = np.tanh(beta*eta)/np.tanh(beta)
    else:
        y = eta
    z = np.linspace(-0.5*np.pi, 0.5*np.pi, nz, endpoint=False)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    return x, y, z, points


def main():
    args = parse_arguments()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    restart_files = sorted(glob.glob(args.pattern))
    if not restart_files:
        raise RuntimeError("No checkpoint files match %r" % args.pattern)

    if min(args.nx, args.ny, args.nz) < 2:
        raise ValueError("nx, ny, and nz must all be at least two")

    if rank == 0:
        os.makedirs(args.output, exist_ok=True)
        x, y, z, points = channel_grid(
            args.nx, args.ny, args.nz, args.beta)
    else:
        x = y = z = points = None

    comm.Barrier()

    mesh = Mesh(comm, create_connectivity=False)
    field = FieldRegistry(comm)
    pynekread(restart_files[0], comm, data_dtype=np.double,
              msh=mesh, fld=field)

    if field.scal_fields < 3:
        raise RuntimeError(
            "Expected temperature, tls, and cls (s0, s1, s2); found %d scalars"
            % field.scal_fields)

    probes = Probes(
        comm,
        probes=points,
        msh=mesh,
        write_coords=False,
        point_interpolator_type="multiple_point_legendre_numpy",
        max_pts=args.max_points,
        find_points_comm_pattern="point_to_point",
    )

    names = ["u", "v", "w", "p", "cls"]

    for file_index, restart_file in enumerate(restart_files):
        if file_index:
            pynekread(restart_file, comm, data_dtype=np.double,
                      fld=field, overwrite_fld=True)

        if field.scal_fields < 3:
            raise RuntimeError("Checkpoint %s does not contain cls=s2"
                               % restart_file)

        fields = [
            field.registry["u"],
            field.registry["v"],
            field.registry["w"],
            field.registry["p"],
            field.registry["s2"],
        ]

        probes.interpolate_from_field_list(
            field.t,
            fields,
            comm,
            write_data=False,
            field_names=names,
        )

        if rank == 0:
            values = probes.interpolated_fields[:, 1:]
            output = {
                "time": np.array(field.t),
                "x": x,
                "y": y,
                "z": z,
            }
            shape = (args.nx, args.ny, args.nz)
            for column, name in enumerate(names):
                output[name] = values[:, column].reshape(shape)

            output_file = os.path.join(
                args.output, os.path.basename(restart_file) + ".npz")
            np.savez(output_file, **output)
            print("Wrote %s at t=%g" % (output_file, field.t))

    if rank == 0:
        point_count = args.nx*args.ny*args.nz
        gib = point_count*len(names)*8/1024**3
        print("Grid: %d x %d x %d" % (args.nx, args.ny, args.nz))
        print("Fields: %s" % ", ".join(names))
        print("Approximate field payload per snapshot: %.2f GiB" % gib)


if __name__ == "__main__":
    start = time.time()
    main()
    if MPI.COMM_WORLD.Get_rank() == 0:
        print("--- interpolation completed in %.2f s ---" %
              (time.time() - start))
