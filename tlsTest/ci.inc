#include <math.h>

static int ciMode = 0;

#define EPS_PROFILE_UNIFORM 5.45e-06;
#define EPS_PROFILE_NONUNIFORM 1.39e-05;
#define EPS_EIKONAL_UNIFORM 2.03e-04;
#define EPS_EIKONAL_NONUNIFORM 6.25e-04;
#define EPS_PROFILE_BCFIX 1.22e-04;
#define EPS_EIKONAL_BCFIX 4.68e-05;


void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");

  if (ciMode == 1 || ciMode == 2) {
    options.setArgs("NUMBER TIMESTEPS", "1000");
  }
}

void ciTestErrors(double time, int tstep, const occa::memory& o_tls)
{
  auto mesh = nrs->meshV;

  // test #1 - verify solution against exact signed-distance function
  auto o_tlsExact = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);

  if (ciMode == 5 || ciMode == 6) {
    udficLine(mesh->Nlocal, mesh->o_x, mesh->o_y, o_tlsExact);
  } else {
    udficCircle(mesh->Nlocal, mesh->o_x, mesh->o_y, o_tlsExact);
  }

  auto o_err_exact = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_tls, -1.0, o_tlsExact, o_err_exact);
  const auto L1norm_exact = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err_exact, platform->comm.mpiComm());

  // test #2 - check Eikonal error e.g. |grad(phi)| = 1.0
  auto o_grad = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * nrs->fluid->fieldOffset);

  launchKernel("core-gradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               nrs->fluid->fieldOffset,
               o_tls,
               o_grad);
  oogs::startFinish(o_grad, mesh->dim, nrs->fluid->fieldOffset, ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmyVector(mesh->Nlocal, nrs->fluid->fieldOffset, 0, 1.0, mesh->o_invLMM, o_grad);

  auto o_err_eik = platform->device.malloc<dfloat>(mesh->Nlocal);

  eikonalError(mesh->Nlocal,
               nrs->fluid->fieldOffset,
               o_grad,
               o_err_eik);

  const auto L1norm_eik =
    platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err_eik, platform->comm.mpiComm());

  dfloat eps, epsEikonal;
  if (ciMode == 1) {
    eps = EPS_PROFILE_UNIFORM;
    epsEikonal = EPS_EIKONAL_UNIFORM;
  } else if (ciMode == 2) {
    eps = EPS_PROFILE_NONUNIFORM;
    epsEikonal = EPS_EIKONAL_NONUNIFORM;
  } else if (ciMode == 3) {
    eps = EPS_PROFILE_UNIFORM;
    epsEikonal = EPS_EIKONAL_UNIFORM;
  } else if (ciMode == 4) {
    eps = EPS_PROFILE_NONUNIFORM;
    epsEikonal = EPS_EIKONAL_NONUNIFORM;
  } else if (ciMode == 5) {
    eps = EPS_PROFILE_BCFIX;
    epsEikonal = EPS_EIKONAL_BCFIX;
  } else if (ciMode == 6) {
    eps = EPS_PROFILE_BCFIX;
    epsEikonal = 10*EPS_EIKONAL_BCFIX;
  }

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("TLSR v/s Exact Signed-Distance Field... L1-Err norm: " + to_string_f(L1norm_exact), L1norm_exact < eps);
    CiEvalTest("TLSR Eikonal Residual... L1-Err norm: " + to_string_f(L1norm_eik), L1norm_eik < epsEikonal);
  }
}
