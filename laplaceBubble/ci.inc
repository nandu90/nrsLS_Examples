#include <math.h>

static int ciMode = 0;

#define EPS_deltaP 3e-02

void ciSetup(MPI_Comm comm, setupAide &options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");
}

void ciTestErrors(double time, int tstep, dfloat deltaPExact)
{
  auto mesh = nrs->meshV;

  const dlong Nlocal = mesh->Nlocal;

  auto o_insideMask = platform->deviceMemoryPool.reserve<dfloat>(Nlocal);
  auto o_outsideMask = platform->deviceMemoryPool.reserve<dfloat>(Nlocal);

  clsPressureMasks(Nlocal,
                   0.001, // interface threshold
                   nrs->scalar->o_solution("cls"),
                   o_insideMask,
                   o_outsideMask);
  
  auto o_pInside = platform->deviceMemoryPool.reserve<dfloat>(Nlocal);
  auto o_pOutside = platform->deviceMemoryPool.reserve<dfloat>(Nlocal);

  maskedPressure(Nlocal, nrs->fluid->o_P, o_insideMask, o_pInside);
  maskedPressure(Nlocal, nrs->fluid->o_P, o_outsideMask, o_pOutside);

  const dfloat pInsideIntegral = platform->linAlg->innerProd(Nlocal, mesh->o_Jw, o_pInside, platform->comm.mpiComm());
  const dfloat pOutsideIntegral = platform->linAlg->innerProd(Nlocal, mesh->o_Jw, o_pOutside, platform->comm.mpiComm());

  const dfloat insideVol = platform->linAlg->innerProd(Nlocal, mesh->o_Jw, o_insideMask, platform->comm.mpiComm());
  const dfloat outsideVol = platform->linAlg->innerProd(Nlocal, mesh->o_Jw, o_outsideMask, platform->comm.mpiComm());

  const dfloat pInsideAvg = pInsideIntegral / insideVol;
  const dfloat pOutsideAvg = pOutsideIntegral / outsideVol;

  const dfloat deltaP = pInsideAvg - pOutsideAvg;
  const dfloat deltaPErr = std::abs(deltaP - deltaPExact);

  if (platform->comm.mpiRank() == 0) {
    std::string msg =
      "Bubble Pressure Jump...\n"
      "  p_inside      : " + to_string_f(pInsideAvg) + "\n" +
      "  p_outside     : " + to_string_f(pOutsideAvg) + "\n" +
      "  deltaP        : " + to_string_f(deltaP) + "\n" +
      "  exact deltaP  : " + to_string_f(deltaPExact) + "\n" +
      "  abs error     : " + to_string_f(deltaPErr);

    CiEvalTest(msg, deltaPErr < EPS_deltaP);
  }
}
