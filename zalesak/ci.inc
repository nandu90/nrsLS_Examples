#include <math.h>

static int ciMode = 0;

#define EPS 2.34e-5

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");

  options.setArgs("END TIME", to_string_f(2.0));
  options.setArgs("DT", to_string_f(1e-04));
}

void ciTestErrors(double time, int tstep, occa::memory& o_exact, occa::memory& o_phi)
{
  auto mesh = nrs->meshV;
  auto &scalar = nrs->scalar;

  auto o_err = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_phi, -1.0, o_exact, o_err);
  const auto L1norm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("L1-Err norm: " + to_string_f(L1norm), L1norm < EPS);
  }
}
