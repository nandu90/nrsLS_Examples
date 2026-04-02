#include <math.h>

static int ciMode = 0;
dfloat EPS_PROFILE_UNIFORM = 6.0e-06;
dfloat EPS_PROFILE_NONUNIFORM = 2.0e-05;
dfloat EPS_EIKONAL_UNIFORM = 2.1e-04;
dfloat EPS_EIKONAL_NONUNIFORM = 6.5e-04;


void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");

  if (ciMode == 1 || ciMode == 2) {
    options.setArgs("NUMBER TIMESTEPS", "1000");
  }
}

void ciTestErrors(nrs_t *nrs, double time, int tstep)
{
  dfloat eps;
  if (ciMode == 1) {
    eps = EPS_PROFILE_UNIFORM;
  } else if (ciMode == 2) {
    eps = EPS_PROFILE_NONUNIFORM;
  }

  auto mesh = nrs->meshV;
  auto &scalar = nrs->scalar;

  auto o_exact = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);

  udfic(mesh->Nlocal, mesh->o_x, mesh->o_y, o_exact);

  auto o_phi = scalar->o_solution("tls");

  auto o_err = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_phi, -1.0, o_exact, o_err);
  const auto L1norm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("Scalar Solver v/s Exact Solution... L1-Err norm: " + to_string_f(L1norm), L1norm < eps);
  }
}

void ciTestErrors(lvlSet_t *ls, double time, int tstep)
{
  dfloat eps, epsEikonal;
  if (ciMode == 3) {
    eps = EPS_PROFILE_UNIFORM;
    epsEikonal = EPS_EIKONAL_UNIFORM;
  } else if (ciMode == 4) {
    eps = EPS_PROFILE_NONUNIFORM;
    epsEikonal = EPS_EIKONAL_NONUNIFORM;
  }

  auto mesh = ls->meshV;

  // test #1 - verify solution against exact signed-distance function
  auto o_exact = platform->deviceMemoryPool.reserve<dfloat>(ls->fieldOffset());

  udfic(mesh->Nlocal, mesh->o_x, mesh->o_y, o_exact);

  auto o_phi = ls->o_S;

  auto o_err_exact = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_phi, -1.0, o_exact, o_err_exact);
  const auto L1norm_exact = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err_exact, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("Level-Set Solver v/s Exact Solution... L1-Err norm: " + to_string_f(L1norm_exact), L1norm_exact < eps);
  }

  // test #2 - checdk Eikonal error e.g. |grad(phi)| = 1.0
  auto o_grad = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * ls->vFieldOffset);

  launchKernel("core-gradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               ls->vFieldOffset,
               ls->o_S,
               o_grad);
  oogs::startFinish(o_grad, mesh->dim, ls->vFieldOffset, ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmyVector(mesh->Nlocal, ls->vFieldOffset, 0, 1.0, mesh->o_invLMM, o_grad);

  auto o_err_eik = platform->device.malloc<dfloat>(mesh->Nlocal);

  eikonalError(mesh->Nlocal,
               ls->vFieldOffset,
               o_grad,
               o_err_eik);

  const auto L1norm_eik =
    platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err_eik, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("Level-Set Solver Eikonal Residual (|grad(phi)| - 1)... L1-Err norm: " + to_string_f(L1norm_eik), L1norm_eik < epsEikonal);
  }

}
