#include <math.h>

static int ciMode = 0;

#define EPS0 1.1e-3
#define EPS1 4.0e-5

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");
}

void ciTestErrors(nrs_t *nrs, double time, int tstep)
{
  auto mesh = nrs->meshV;
  auto &scalar = nrs->scalar;

  auto o_exact = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);

  exact(mesh->Nlocal, mesh->o_x, mesh->o_y, o_exact);

  auto o_phi = scalar->o_solution("phi");

  auto o_err = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_phi, -1.0, o_exact, o_err);
  const auto L1norm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("Scalar Solver v/s Exact Solution... L1-Err norm: " + to_string_f(L1norm), L1norm < EPS0);
  }
}

void ciTestErrors(lvlSet_t *ls, double time, int tstep)
{
  auto mesh = ls->meshV;

  auto o_exact = platform->deviceMemoryPool.reserve<dfloat>(ls->fieldOffset());

  exact(mesh->Nlocal, mesh->o_x, mesh->o_y, o_exact);

  auto o_phi = ls->o_S;

  auto o_err = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_phi, -1.0, o_exact, o_err);
  const auto L1norm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("Level-Set Solver v/s Exact Solution... L1-Err norm: " + to_string_f(L1norm), L1norm < EPS0);
  }
}

void ciTestErrors(nrs_t *nrs, lvlSet_t *ls, double time, int tstep)
{
  auto mesh = nrs->meshV;

  auto o_scalar = nrs->scalar->o_solution("phi");
  auto o_lvlset = ls->o_S;

  auto o_err = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_scalar, -1.0, o_lvlset, o_err);
  const auto L1norm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("Scalar Solver v/s Level-Set Solver L1-Err norm: " + to_string_f(L1norm), L1norm < EPS1);
  }
}
