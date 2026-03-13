#include <math.h>

static int ciMode = 0;

#define EPS 5e-4

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "1");

  options.setArgs("SCALAR00 SOLVER TOLERANCE", std::string("1e-14"));
}

// void ciTestErrors(nrs_t *nrs, double time, int tstep)
// {
//   if (!nrs->lastStep) {
//     return;
//   }

//   auto mesh = nrs->meshV;
//   auto &scalar = nrs->scalar;

//   auto o_exact = platform->deviceMemoryPool.reserve<dfloat>(nrs->fluid->fieldOffset);

//   exact(mesh->Nlocal, mesh->o_x, mesh->o_y, o_exact);

//   auto o_phi = scalar->o_solution("phi");

//   auto o_err = platform->device.malloc<dfloat>(mesh->Nlocal);

//   platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_phi, -1.0, o_exact, o_err);
//   const auto L1norm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

//   if (platform->comm.mpiRank() == 0) {
//     CiEvalTest("Scalar Solver v/s Exact Solution... L1-Err norm: " + to_string_f(L1norm), L1norm < EPS);
//   }
// }

void ciTestErrors(ls_t *tlsr, double time, int tstep)
{
  auto mesh = tlsr->meshV;

  // test #1 - verify solution against exact signed-distance function
  auto o_exact = platform->deviceMemoryPool.reserve<dfloat>(tlsr->fieldOffset());

  exact(mesh->Nlocal, mesh->o_x, mesh->o_y, o_exact);

  auto o_phi = tlsr->o_solution("tlsr");

  auto o_err_exact = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_phi, -1.0, o_exact, o_err_exact);
  const auto L1norm_exact = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err_exact, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("Level-Set Solver v/s Exact Solution... L1-Err norm: " + to_string_f(L1norm_exact), L1norm_exact < EPS);
  }

  // test #2 - checdk Eikonal error e.g. |grad(phi)| = 1.0
  auto o_grad = platform->deviceMemoryPool.reserve<dfloat>(mesh->dim * tlsr->vFieldOffset);

  launchKernel("core-gradientVolumeHex3D",
               mesh->Nelements,
               mesh->o_vgeo,
               mesh->o_D,
               tlsr->vFieldOffset,
               tlsr->o_solution("tlsr"),
               o_grad);
  oogs::startFinish(o_grad, mesh->dim, tlsr->vFieldOffset, ogsDfloat, ogsAdd, mesh->oogs);
  platform->linAlg->axmyVector(mesh->Nlocal, tlsr->vFieldOffset, 0, 1.0, mesh->o_invLMM, o_grad);

  auto o_err_eik = platform->device.malloc<dfloat>(mesh->Nlocal);

  eikonalError(mesh->Nlocal,
               tlsr->vFieldOffset,
               o_grad,
               o_err_eik);

  const auto L1norm_eik =
    platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err_eik, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("TLSR Eikonal Residual (|grad(phi)| - 1)... L1-Err norm: " + to_string_f(L1norm_eik), L1norm_eik < EPS);
  }

}

// void ciTestErrors(nrs_t *nrs, ls_t *tlsr, double time, int tstep)
// {
//   auto mesh = nrs->meshV;

//   auto o_scalar = nrs->scalar->o_solution("phi");
//   auto o_lvlset = tlsr->o_solution("tlsr");

//   auto o_err = platform->device.malloc<dfloat>(mesh->Nlocal);

//   platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_scalar, -1.0, o_lvlset, o_err);
//   const auto L1norm = platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err, platform->comm.mpiComm());

//   if (platform->comm.mpiRank() == 0) {
//     CiEvalTest("Scalar Solver v/s Level-Set Solver L1-Err norm: " + to_string_f(L1norm), L1norm < EPS);
//   }
// }
