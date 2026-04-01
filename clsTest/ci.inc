#include <math.h>

static int ciMode = 0;

#define EPS_PROFILE 5.0e-3
#define EPS_MASS 5.0e-07

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");
}

void ciTestErrorsCLS(lvlSet_t *ls, double time, int tstep, const dfloat clsMass0)
{
  auto mesh = ls->meshV;

  // test #1 - verify CLSR solution against exact conservative profile
  auto o_phiExact = platform->deviceMemoryPool.reserve<dfloat>(ls->fieldOffset());
  auto o_clsExact = platform->deviceMemoryPool.reserve<dfloat>(ls->fieldOffset());

  // initialize exact signed-distance field for the circular interface
  udfic(mesh->Nlocal, mesh->o_x, mesh->o_y, o_phiExact);

  // map to conservative profile using solver's internal interface width (epsin = 0.0)
  lvlSet::initHeaviside(o_phiExact, o_clsExact, 0.0);

  auto o_cls = ls->o_S;

  auto o_err_exact = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_cls, -1.0, o_clsExact, o_err_exact);
  const auto L1norm_exact =
    platform->linAlg->weightedNorm1(mesh->Nlocal, mesh->o_LMM, o_err_exact, platform->comm.mpiComm());

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("CLSR v/s Exact Conservative Profile... L1-Err norm: " + to_string_f(L1norm_exact),
               L1norm_exact < EPS_PROFILE);
  }

  // test #2 - verify conservative property (total CLS mass should remain invariant)
  const auto clsMass =
    platform->linAlg->innerProd(mesh->Nlocal, mesh->o_LMM, o_cls, platform->comm.mpiComm());

  const dfloat relMassErr =
    std::abs(clsMass - clsMass0) / (std::abs(clsMass0) + 1e-16);

  if (platform->comm.mpiRank() == 0) {
    std::string msg =
      "CLSR Mass Conservation...\n"
      "  initial mass : " + to_string_f(clsMass0) + "\n" +
      "  final mass   : " + to_string_f(clsMass) + "\n" +
      "  rel error    : " + to_string_f(relMassErr);

    CiEvalTest(msg, relMassErr < EPS_MASS);
  }
}
