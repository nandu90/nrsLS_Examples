from mpi4py import MPI
import glob
import numpy as np
import os
import time

from pysemtools.io.ppymech.neksuite import pynekread
from pysemtools.datatypes.msh import Mesh
from pysemtools.datatypes.field import FieldRegistry
from pysemtools.interpolation.probes import Probes


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    restartFiles = sorted(glob.glob("*0.f00*"))
    outputDir = "interpolatedData"
    n = 256

    if len(restartFiles) == 0:
        raise RuntimeError("No restart files matching *0.f00* were found")

    if rank == 0:
        os.makedirs(outputDir, exist_ok=True)

    comm.Barrier()

    mesh = Mesh(comm, create_connectivity=False)
    field = FieldRegistry(comm)

    pynekread(restartFiles[0], comm, data_dtype=np.double,
              msh=mesh, fld=field)

    if field.scal_fields != 2:
        raise RuntimeError("Expected the tls and cls scalar fields")

    if rank == 0:
        x1d = np.linspace(-np.pi, np.pi, n, endpoint=False)
        y1d = np.linspace(-np.pi, np.pi, n, endpoint=False)
        z1d = np.linspace(-np.pi, np.pi, n, endpoint=False)

        x, y, z = np.meshgrid(x1d, y1d, z1d, indexing="ij")
        xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    else:
        xyz = None

    probes = Probes(
        comm,
        probes=xyz,
        msh=mesh,
        write_coords=False,
        point_interpolator_type="multiple_point_legendre_numpy",
        max_pts=256,
        find_points_comm_pattern="point_to_point",
    )

    for fileIndex, restartFile in enumerate(restartFiles):
        if fileIndex > 0:
            pynekread(restartFile, comm, data_dtype=np.double,
                      fld=field, overwrite_fld=True)

        if field.scal_fields != 2:
            raise RuntimeError("Expected the tls and cls scalar fields")

        fieldNames = ["u", "v", "w", "p", "tls", "cls"]
        fieldList = [
            field.registry["u"],
            field.registry["v"],
            field.registry["w"],
            field.registry["p"],
            field.registry["s0"],
            field.registry["s1"],
        ]

        probes.interpolate_from_field_list(
            field.t,
            fieldList,
            comm,
            write_data=False,
            field_names=fieldNames,
        )

        if rank == 0:
            interpolated = probes.interpolated_fields[:, 1:]

            output = {
                "time": np.array(field.t),
                "x": x1d,
                "y": y1d,
                "z": z1d,
            }

            for i, name in enumerate(fieldNames):
                output[name] = interpolated[:, i].reshape(n, n, n)

            outputFile = os.path.join(
                outputDir, os.path.basename(restartFile) + ".npz")
            np.savez(outputFile, **output)

            print("Wrote %s" % outputFile)

    if rank == 0:
        print("Grid: %d x %d x %d" % (n, n, n))
        print("Fields: %s" % ", ".join(fieldNames))

    return


if __name__ == "__main__":
    starttime = time.time()
    main()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("--- Code ran in %s seconds ---" %
              (time.time() - starttime))
