#include <math.h>

static int ciMode = 0;

#define EPS 1.06e-3

void ciSetup(MPI_Comm comm, setupAide &options)
{
  // options.setArgs("SCALAR00 SOLVER TOLERANCE", std::string("1e-14"));
}

void ciTestErrorsCLS(lvlSet_t *ls, double time, int tstep, const dfloat clsMass0)
{
  auto mesh = ls->meshV;

  // test #1 - verify CLSR solution against exact conservative profile
  // exact conservative profile = Heaviside(exact signed-distance, internal interfaceWidth)

  auto o_phiExact = platform->deviceMemoryPool.reserve<dfloat>(ls->fieldOffset());
  auto o_clsExact = platform->deviceMemoryPool.reserve<dfloat>(ls->fieldOffset());

  udfic(mesh->Nlocal, mesh->o_x, mesh->o_y, o_phiExact);
  lvlSet::initHeaviside(o_phiExact, o_clsExact, 0.0);

  auto o_cls = ls->o_S;

  auto o_err_exact = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_cls, -1.0, o_clsExact, o_err_exact);
  const auto L1norm_exact =
    platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err_exact, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("CLSR v/s Exact Conservative Profile... L1-Err norm: " + to_string_f(L1norm_exact),
               L1norm_exact < EPS);
  }

  // test #2 - mass conservation
  const auto clsMass =
    platform->linAlg->innerProd(mesh->Nlocal, mesh->o_LMM, o_cls, platform->comm.mpiComm());

  const dfloat relMassErr = std::abs(clsMass - clsMass0) / (std::abs(clsMass0) + 1e-16);

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("CLSR Mass Conservation... RelErr: " + to_string_f(relMassErr),
               relMassErr < EPS);
  }
}
