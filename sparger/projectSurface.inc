/*
  Project selected boundary nodes onto an exact circular cylinder.

  Include kernels/projectSurface.oudf before this file in the case UDF.

  The axis vector is normal to a cylinder cross-section. For example,
  axis = (0, 1, 0) creates a cylinder whose axis is parallel to y. The center
  is any point on the cylinder axis. Only boundary nodes whose current radial
  distance differs from targetRadius by no more than tolerance are moved.

  Example:

    #include "/home/nico/usrRS/projectSurface.inc"

    #ifdef __okl__

    #include "/home/nico/usrRS/kernels/projectSurface.oudf"

    #endif

    void UDF_Setup()
    {
      // Project boundary ID 3 onto a z-aligned cylinder.
      projectSurfaceToCylinder(3,
                               0.0, 0.0, 1.0,
                               centerX, centerY, centerZ,
                               radius, tolerance);
    }
*/

#ifndef USRRS_PROJECT_SURFACE_INC
#define USRRS_PROJECT_SURFACE_INC

#include <cmath>

void projectSurfaceToCylinder(mesh_t *mesh,
                              const dlong bID,
                              const dfloat axisX,
                              const dfloat axisY,
                              const dfloat axisZ,
                              const dfloat centerX,
                              const dfloat centerY,
                              const dfloat centerZ,
                              const dfloat targetRadius,
                              const dfloat tolerance,
                              const bool updateGeometry = true)
{
  nekrsCheck(mesh == nullptr,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "projectSurfaceToCylinder: mesh is null");
  nekrsCheck(bID <= 0,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "projectSurfaceToCylinder: bID must be positive");
  nekrsCheck(targetRadius <= 0.0,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "projectSurfaceToCylinder: targetRadius must be positive");
  nekrsCheck(tolerance < 0.0,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "projectSurfaceToCylinder: tolerance must be nonnegative");

  const dfloat axisMagnitude = sqrt(axisX*axisX + axisY*axisY + axisZ*axisZ);

  nekrsCheck(axisMagnitude <= 1.0e-14,
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s\n",
             "projectSurfaceToCylinder: cylinder axis has zero length");

  const dfloat unitAxisX = axisX/axisMagnitude;
  const dfloat unitAxisY = axisY/axisMagnitude;
  const dfloat unitAxisZ = axisZ/axisMagnitude;

  projectSurface(mesh->Nelements,
                 bID,
                 unitAxisX,
                 unitAxisY,
                 unitAxisZ,
                 centerX,
                 centerY,
                 centerZ,
                 targetRadius,
                 tolerance,
                 mesh->o_vmapM,
                 mesh->o_EToB,
                 mesh->o_x,
                 mesh->o_y,
                 mesh->o_z);

  // Coordinate changes invalidate volume and surface geometric factors.
  if (updateGeometry) {
    mesh->update();
  }
}

void projectSurfaceToCylinder(const dlong bID,
                              const dfloat axisX,
                              const dfloat axisY,
                              const dfloat axisZ,
                              const dfloat centerX,
                              const dfloat centerY,
                              const dfloat centerZ,
                              const dfloat targetRadius,
                              const dfloat tolerance,
                              const bool updateGeometry = true)
{
  projectSurfaceToCylinder(nrs->meshV,
                           bID,
                           axisX,
                           axisY,
                           axisZ,
                           centerX,
                           centerY,
                           centerZ,
                           targetRadius,
                           tolerance,
                           updateGeometry);
}

#endif
