#include <math.h>

static int ciMode = 0;

#define EPS_CLS 4.74e-6
#define EPS_TLS 3.59e-6

static occa::memory o_phiExact;

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");

  options.setArgs("DT", to_string_f(2e-04));
}

void ciTestErrors(double time, int tstep, occa::memory& o_psi, occa::memory& o_phi)
{
  auto mesh = nrs->meshV;

  if(!o_phiExact.isInitialized()) {
    //store the TLSR field at zeroth time step
    o_phiExact = platform->device.malloc<dfloat>(mesh->Nlocal);
    o_phi.copyTo(o_phiExact, mesh->Nlocal);
  }

  if(!nrs->lastStep)
    return;

  auto o_err = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

  auto o_signedDistance = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  auto o_psiExact = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  tlsic(mesh->Nlocal, mesh->o_x, mesh->o_y, o_signedDistance);
  lvlSet::initHeaviside(o_signedDistance, o_psiExact);
  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_psi, -1.0, o_psiExact, o_err);
  const auto normPsi = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_phi, -1.0, o_phiExact, o_err);
  const auto normPhi = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("L1-Err norm CLS: " + to_string_f(normPsi) + " TLS: " + to_string_f(normPhi), (normPsi < EPS_CLS && normPhi < EPS_TLS));
  }
}
