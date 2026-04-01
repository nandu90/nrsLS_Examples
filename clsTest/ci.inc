#include <math.h>

static int ciMode = 0;

#define EPS_UNIFORM 4.93e-3
#define EPS_VOL_UNIFORM 7.66e-03
#define EPS_MIN 4.37e-3
#define EPS_VOL_MIN 6.75e-03
#define EPS_MEAN 4.37e-3
#define EPS_VOL_MEAN 6.75e-03
#define EPS_MAX 4.37e-3
#define EPS_VOL_MAX 6.75e-03

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");
  if(ciMode == 2) {
    options.setArgs("LVLSET INTERFACE WIDTH MESH PARAM","MIN");
  }
  else if(ciMode == 3) {
    options.setArgs("LVLSET INTERFACE WIDTH MESH PARAM","MEAN");
  }
  else if(ciMode == 4) {
    options.setArgs("LVLSET INTERFACE WIDTH MESH PARAM","MAX");
  }
}

void ciTestErrors(double time, int tstep, const occa::memory& o_cls)
{
  auto mesh = nrs->meshV;

  // test #1 - verify CLSR solution against exact conservative profile
  auto o_clsExact = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  {
    auto o_phiExact = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    // initialize exact signed-distance field for the circular interface
    udfic(mesh->Nlocal, mesh->o_x, mesh->o_y, o_phiExact);

    // map to conservative profile using solver's internal interface width
    lvlSet::initHeaviside(o_phiExact, o_clsExact);
  }
  auto clsVolExact = platform->linAlg->innerProd(mesh->Nlocal,
                                                 mesh->o_LMM,
                                                 o_clsExact,
                                                 platform->comm.mpiComm());

  auto o_err_exact = platform->device.malloc<dfloat>(mesh->Nlocal);

  platform->linAlg->axpbyz(mesh->Nlocal, 1.0, o_cls, -1.0, o_clsExact, o_err_exact);

  const auto L1norm_exact = platform->linAlg->weightedNorm1(mesh->Nlocal, 
                                                            mesh->o_LMM,
                                                            o_err_exact,
                                                            platform->comm.mpiComm());

  const auto clsVol = platform->linAlg->innerProd(mesh->Nlocal, 
                                                  mesh->o_LMM,
                                                  o_cls,
                                                  platform->comm.mpiComm());

  dfloat eps, epsVol;
  if(ciMode == 1) {
    eps = EPS_UNIFORM;
    epsVol = EPS_VOL_UNIFORM;
  } else if (ciMode == 2) {
    eps = EPS_MIN;
    epsVol = EPS_VOL_MIN;
  } else if (ciMode == 3) {
    eps = EPS_MEAN;
    epsVol = EPS_VOL_MEAN;
  } else if (ciMode == 4) {
    eps = EPS_MAX;
    epsVol = EPS_VOL_MAX;
  }

  const dfloat relVolErr = std::abs(clsVol - clsVolExact) / (std::abs(clsVolExact) + 1e-16);

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("CLSR v/s Exact Conservative Profile... L1-Err norm: " + to_string_f(L1norm_exact),
               L1norm_exact < eps);

    std::string msg =
      "CLSR Mass Conservation...\n"
      "  exact volume : " + to_string_f(clsVolExact) + "\n" +
      "  final volume   : " + to_string_f(clsVol) + "\n" +
      "  rel error    : " + to_string_f(relVolErr);

    CiEvalTest(msg, relVolErr < epsVol);
  }
}
