#include <math.h>

static int ciMode = 0;

#define EPS 1.06e-3

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");

  options.setArgs("SCALAR00 SOLVER TOLERANCE", std::string("1e-8"));
}

void ciTestErrors(nrs_t *nrs, double time, int tstep)
{
  if (!nrs->lastStep) {
    return;
  }

  auto mesh = nrs->meshV;
  auto &scalar = nrs->scalar;

  auto o_exact = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);

  exact(mesh->Nlocal, mesh->o_x, mesh->o_y, o_exact);

  auto o_phi = scalar->o_solution("phi");

  auto o_err = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_phi, -1.0, o_exact, o_err);
  const auto L1norm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("L1-Err norm: " + to_string_f(L1norm), L1norm < EPS);
  }
}
