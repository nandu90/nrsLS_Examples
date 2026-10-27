#ifndef POST_PROCESSING_INC
#define POST_PROCESSING_INC

#include "yp.inc"
#include "lvlSetDiagnostics.inc"
#include "pointInterpolation.hpp"
#include <algorithm>

static const bool collectAvg = true;

static FILE *bubblyFile = nullptr;
static FILE *budgetProfileFile = nullptr;
static bool twoPhaseMonitorsEnabled = false;
static constexpr int numberProfilePoints = 100;
static int numberNativeProfilePoints = 0;
static dfloat referenceGasVolume = 0.0;
static deviceMemory<dfloat> o_surfaceTensionAcceleration;
static deviceMemory<dfloat> o_budgetDiagnosticFields;
static deviceMemory<dfloat> o_profileInstant;
static deviceMemory<dfloat> o_profileAverage;
static deviceMemory<dlong> o_profilePlaneIndex;
static deviceMemory<dfloat> o_profilePlaneWeights;
static deviceMemory<dfloat> o_nativePlaneAverage;
static pointInterpolation_t *profileInterpolator = nullptr;
static std::vector<dfloat> profileY;
static dfloat profileAveragingTime = 0.0;
static dfloat previousProfileTime = 0.0;
static bool firstProfileSample = true;

enum BudgetDiagnosticField
{
  meanU = 0,
  meanV,
  meanW,
  meanRho,
  meanMu,
  meanCls,
  capillaryForceX,
  capillaryForceY,
  capillaryForceZ,
  capillaryPower,
  viscousDissipation,
  pressureGradientPower,
  gravityPower,
  viscousStress11,
  viscousStress12,
  viscousStress13,
  viscousStress22,
  viscousStress23,
  viscousStress33,
  velocityCapillaryX,
  velocityCapillaryY,
  velocityCapillaryZ,
  densityVelocityX,
  densityVelocityY,
  densityVelocityZ,
  numberOfBudgetDiagnosticFields
};

static const std::array<const char *, numberOfBudgetDiagnosticFields>
    budgetFieldNames = {
        "u", "v", "w", "rho", "mu", "cls",
        "fSigmaX", "fSigmaY", "fSigmaZ", "surfaceTensionPower",
        "viscousDissipation", "pressureGradientPower", "gravityPower",
        "tau11", "tau12", "tau13", "tau22", "tau23", "tau33",
        "uFSigmaX", "vFSigmaY", "wFSigmaZ", "rhoU", "rhoV", "rhoW"};

void setupCoordinateBasedProfiles()
{
  auto mesh = nrs->meshV;
  const auto comm = platform->comm.mpiComm();
  const int rank = platform->comm.mpiRank();

  std::vector<dfloat> localY(mesh->Nlocal);
  mesh->o_y.copyTo(localY.data(), mesh->Nlocal);

  const dfloat ymin = platform->linAlg->min(
      mesh->Nlocal, mesh->o_y, comm);
  const dfloat ymax = platform->linAlg->max(
      mesh->Nlocal, mesh->o_y, comm);
  const dfloat yTolerance =
      1.0e-10*std::max<dfloat>(1.0, fabs(ymax - ymin));

  // Every node at the same physical y location receives the same plane key.
  // This does not depend on global element numbering and therefore remains
  // valid for both ordinary box meshes and runtime h-refinement.
  std::vector<hlong> planeId(mesh->Nlocal);
  std::vector<hlong> localPlaneIds;
  localPlaneIds.reserve(mesh->Nlocal);
  for (dlong node = 0; node < mesh->Nlocal; ++node) {
    const hlong id =
        static_cast<hlong>(llround((localY[node] - ymin)/yTolerance)) + 1;
    planeId[node] = id;
    localPlaneIds.push_back(id);
  }

  std::sort(localPlaneIds.begin(), localPlaneIds.end());
  localPlaneIds.erase(
      std::unique(localPlaneIds.begin(), localPlaneIds.end()),
      localPlaneIds.end());
  const int size = platform->comm.mpiCommSize();
  const int localCount = static_cast<int>(localPlaneIds.size());
  std::vector<int> counts(size, 0);
  MPI_Allgather(&localCount, 1, MPI_INT,
                counts.data(), 1, MPI_INT, comm);
  std::vector<int> displacements(size, 0);
  for (int r = 1; r < size; ++r) {
    displacements[r] = displacements[r - 1] + counts[r - 1];
  }
  const int gatheredCount = displacements.back() + counts.back();
  std::vector<hlong> globalPlaneIds(gatheredCount);
  MPI_Allgatherv(localPlaneIds.data(), localCount, MPI_HLONG,
                 globalPlaneIds.data(), counts.data(), displacements.data(),
                 MPI_HLONG, comm);
  std::sort(globalPlaneIds.begin(), globalPlaneIds.end());
  globalPlaneIds.erase(
      std::unique(globalPlaneIds.begin(), globalPlaneIds.end()),
      globalPlaneIds.end());
  numberNativeProfilePoints = static_cast<int>(globalPlaneIds.size());

  std::vector<dlong> planeIndex(mesh->Nlocal);
  for (dlong node = 0; node < mesh->Nlocal; ++node) {
    planeIndex[node] = std::lower_bound(globalPlaneIds.begin(),
                                        globalPlaneIds.end(),
                                        planeId[node])
                     - globalPlaneIds.begin();
  }
  o_profilePlaneIndex.resize(mesh->Nlocal);
  o_profilePlaneIndex.copyFrom(planeIndex.data(), planeIndex.size());

  // LMM/sum_plane(LMM) supplies the x-z quadrature weight. The wall-normal
  // contribution cancels within a constant-y plane for this channel mesh.
  o_profilePlaneWeights.resize(mesh->Nlocal);
  auto o_planeWeightSum =
      platform->deviceMemoryPool.reserve<dfloat>(numberNativeProfilePoints);
  platform->linAlg->fill(numberNativeProfilePoints, 0.0, o_planeWeightSum);
  accumulateProfilePlaneWeights(mesh->Nlocal,
                                o_profilePlaneIndex,
                                mesh->o_LMM,
                                o_planeWeightSum);
  platform->comm.allreduce(o_planeWeightSum,
                           numberNativeProfilePoints,
                           comm_t::op::sum,
                           comm);
  normalizeProfilePlaneWeights(mesh->Nlocal,
                               o_profilePlaneIndex,
                               o_planeWeightSum,
                               mesh->o_LMM,
                               o_profilePlaneWeights);
  o_nativePlaneAverage.resize(
      numberOfBudgetDiagnosticFields*numberNativeProfilePoints);

  // NekRS performs element location and high-order spectral interpolation of
  // the planar-averaged fields at the fixed wall-normal output locations.
  const dfloat xmin = platform->linAlg->min(
      mesh->Nlocal, mesh->o_x, comm);
  const dfloat xmax = platform->linAlg->max(
      mesh->Nlocal, mesh->o_x, comm);
  const dfloat zmin = platform->linAlg->min(
      mesh->Nlocal, mesh->o_z, comm);
  const dfloat zmax = platform->linAlg->max(
      mesh->Nlocal, mesh->o_z, comm);
  profileY.resize(numberProfilePoints);
  std::vector<dfloat> profileX;
  std::vector<dfloat> interpolationY;
  std::vector<dfloat> profileZ;
  const dfloat pi = acos(-1.0);
  for (int point = 0; point < numberProfilePoints; ++point) {
    const dfloat fraction =
        static_cast<dfloat>(point)/(numberProfilePoints - 1);
    profileY[point] = 0.5*(ymin + ymax)
                    - 0.5*(ymax - ymin)*cos(pi*fraction);
    if (rank == 0) {
      profileX.push_back(0.5*(xmin + xmax));
      interpolationY.push_back(profileY[point]);
      profileZ.push_back(0.5*(zmin + zmax));
    }
  }
  profileInterpolator = new pointInterpolation_t(mesh, comm);
  profileInterpolator->setPoints(profileX, interpolationY, profileZ);
  profileInterpolator->find(pointInterpolation_t::VerbosityLevel::None);

}

void coordinateBasedPlanarAverage(occa::memory& o_fields)
{
  auto mesh = nrs->meshV;
  const auto offset = nrs->fluid->fieldOffset;
  platform->linAlg->fill(o_nativePlaneAverage.size(),
                         0.0,
                         o_nativePlaneAverage);
  accumulateProfilePlaneFields(mesh->Nlocal,
                               offset,
                               numberNativeProfilePoints,
                               numberOfBudgetDiagnosticFields,
                               o_profilePlaneIndex,
                               o_profilePlaneWeights,
                               o_fields,
                               o_nativePlaneAverage);
  platform->comm.allreduce(o_nativePlaneAverage,
                           o_nativePlaneAverage.size(),
                           comm_t::op::sum,
                           platform->comm.mpiComm());
  scatterProfilePlaneFields(mesh->Nlocal,
                            offset,
                            numberNativeProfilePoints,
                            numberOfBudgetDiagnosticFields,
                            o_profilePlaneIndex,
                            o_nativePlaneAverage,
                            o_fields);
}

dfloat activePressureGradient()
{
  auto mesh = nrs->meshV;
  const auto offset = nrs->fluid->fieldOffset;
  auto o_rho = nrs->fluid->o_prop.slice(offset, mesh->Nlocal);
  const dfloat rhoMean = platform->linAlg->innerProd(
      mesh->Nlocal,
      mesh->o_LMM,
      o_rho,
      platform->comm.mpiComm())/mesh->volume;

  if (platform->options.compareArgs("CONSTANT FLOW RATE", "TRUE")) {
    return rhoMean/Fr + nrs->flowRateScaleFactor();
  }

  return 4.0*rhol*Ubulk*pow(Re_tau/Re, 2.0) + rhoMean/Fr;
}

void writeBudgetProfiles(const dfloat time)
{
  if (profileAveragingTime <= 0.0) {
    return;
  }

  std::vector<dfloat> values(
      numberOfBudgetDiagnosticFields*numberProfilePoints);
  o_profileAverage.copyTo(values);

  if (platform->comm.mpiRank() == 0) {
    fprintf(budgetProfileFile, "# intervalEnd %.16e duration %.16e points %d\n",
            time, profileAveragingTime, numberProfilePoints);
    for (int point = 0; point < numberProfilePoints; ++point) {
      fprintf(budgetProfileFile, "%.16e", profileY[point]);
      for (int field = 0; field < numberOfBudgetDiagnosticFields; ++field) {
        fprintf(budgetProfileFile, " %.16e",
                values[field*numberProfilePoints + point]);
      }
      fprintf(budgetProfileFile, "\n");
    }
    fprintf(budgetProfileFile, "\n");
    fflush(budgetProfileFile);
  }

  platform->linAlg->fill(o_profileAverage.size(), 0.0, o_profileAverage);
  profileAveragingTime = 0.0;
}

void openBubblyFile(const bool restartingTwoPhase)
{
  static const char *header =
      "# t alphaG volumeG relativeVolumeError area interfaceThickness "
      "d32 surfaceEnergy kL kG "
      "uLx uLy uLz uGx uGy uGz slipX slipY slipZ slipMagnitude\n";

  if (platform->comm.mpiRank() != 0) {
    return;
  }

  if (restartingTwoPhase) {
    dfloat restartTime = 0.0;
    platform->options.getArgs("START TIME", restartTime);

    std::ifstream input("bubbly.dat");
    std::ofstream output("bubbly.dat.tmp", std::ios::trunc);
    if (!output) {
      fprintf(stderr, "Unable to open bubbly.dat.tmp for writing\n");
      MPI_Abort(platform->comm.mpiComm(), 1);
    }

    output << header;
    bool foundReferenceVolume = false;
    std::string line;
    while (std::getline(input, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }

      std::istringstream values(line);
      dfloat fileTime = 0.0;
      dfloat fileAlphaGas = 0.0;
      dfloat fileGasVolume = 0.0;
      if (values >> fileTime >> fileAlphaGas >> fileGasVolume) {
        if (!foundReferenceVolume && fileGasVolume > 0.0) {
          referenceGasVolume = fileGasVolume;
          foundReferenceVolume = true;
        }
        if (fileTime < restartTime - 1.0e-12) {
          output << line << '\n';
        }
      }
    }
    input.close();
    output.close();

    if (rename("bubbly.dat.tmp", "bubbly.dat") != 0) {
      fprintf(stderr,
              "Unable to replace bubbly.dat after trimming restart history\n");
      MPI_Abort(platform->comm.mpiComm(), 1);
    }
    bubblyFile = fopen("bubbly.dat", "a");
  }
  else {
    bubblyFile = fopen("bubbly.dat", "w");
    if (bubblyFile) {
      fprintf(bubblyFile, "%s", header);
    }
  }

  if (!bubblyFile) {
    fprintf(stderr, "Unable to open bubbly.dat for writing\n");
    MPI_Abort(platform->comm.mpiComm(), 1);
  }
  fflush(bubblyFile);
}

void postProcessingSetup(const bool restarting,
                         const bool initializedBubbles)
{
  auto mesh = nrs->meshV;

  const auto clsField = nrs->scalar->nameToIndex.find("cls");
  nekrsCheck(clsField == nrs->scalar->nameToIndex.end(),
             platform->comm.mpiComm(),
             EXIT_FAILURE,
             "%s",
             "CLS scalar is required by bubblyFlow post-processing");
  twoPhaseMonitorsEnabled = nrs->scalar->compute[clsField->second];

  if (twoPhaseMonitorsEnabled) {
    referenceGasVolume = lsDiagnostic::enclosedVolume();
    const bool restartingTwoPhase = restarting && !initializedBubbles;
    openBubblyFile(restartingTwoPhase);
    MPI_Bcast(&referenceGasVolume,
              1,
              MPI_DFLOAT,
              0,
              platform->comm.mpiComm());
  }

  if (twoPhaseMonitorsEnabled && collectAvg) {
    const auto offset = nrs->fluid->fieldOffset;
    setupCoordinateBasedProfiles();
    o_surfaceTensionAcceleration.resize(nrs->fluid->fieldOffsetSum);
    o_budgetDiagnosticFields.resize(numberOfBudgetDiagnosticFields*offset);
    o_profileInstant.resize(numberOfBudgetDiagnosticFields*numberProfilePoints);
    o_profileAverage.resize(numberOfBudgetDiagnosticFields*numberProfilePoints);
    platform->linAlg->fill(o_profileInstant.size(), 0.0, o_profileInstant);
    platform->linAlg->fill(o_profileAverage.size(), 0.0, o_profileAverage);

    if (platform->comm.mpiRank() == 0) {
      budgetProfileFile = fopen("budgetProfiles.dat", "w");
      if (!budgetProfileFile) {
        fprintf(stderr, "Unable to open budgetProfiles.dat for writing\n");
        MPI_Abort(platform->comm.mpiComm(), 1);
      }
      fprintf(budgetProfileFile, "# y");
      for (const auto *name : budgetFieldNames) {
        fprintf(budgetProfileFile, " %s", name);
      }
      fprintf(budgetProfileFile, "\n");
      fflush(budgetProfileFile);
    }
  }
}

void saveSurfaceTensionForPostProcessing(
    const occa::memory& o_forceAcceleration)
{
  if (twoPhaseMonitorsEnabled && collectAvg) {
    o_surfaceTensionAcceleration.copyFrom(
        o_forceAcceleration, nrs->fluid->fieldOffsetSum);
  }
}

void collectBudgetProfiles(const dfloat time)
{
  if (!twoPhaseMonitorsEnabled || !collectAvg) {
    return;
  }

  auto mesh = nrs->meshV;
  const auto offset = nrs->fluid->fieldOffset;
  const dfloat pressureGradient = activePressureGradient();
  auto o_rho = nrs->fluid->o_prop.slice(1*offset, mesh->Nlocal);
  auto o_mue = nrs->fluid->o_prop.slice(0*offset, mesh->Nlocal);
  auto o_cls = nrs->scalar->o_solution("cls");
  auto o_Sij = nrs->strainRate();

  assembleBudgetDiagnostics(
      mesh->Nlocal,
      offset,
      pressureGradient,
      1.0/Fr,
      nrs->fluid->o_U,
      o_rho,
      o_mue,
      o_cls,
      o_Sij,
      o_surfaceTensionAcceleration,
      o_budgetDiagnosticFields);

  if (nrs->timeStepConverged) {
    coordinateBasedPlanarAverage(o_budgetDiagnosticFields);

    profileInterpolator->eval(numberOfBudgetDiagnosticFields,
                              offset,
                              o_budgetDiagnosticFields,
                              numberProfilePoints,
                              o_profileInstant);

    if (firstProfileSample) {
      previousProfileTime = time;
      firstProfileSample = false;
    }
    const dfloat interval = time - previousProfileTime;
    if (interval > 0.0) {
      const dfloat newAveragingTime = profileAveragingTime + interval;
      const dfloat oldWeight = profileAveragingTime/newAveragingTime;
      const dfloat newWeight = interval/newAveragingTime;
      platform->linAlg->axpby(
          numberOfBudgetDiagnosticFields*numberProfilePoints,
          newWeight,
          o_profileInstant,
          oldWeight,
          o_profileAverage);
      profileAveragingTime = newAveragingTime;
    }
    previousProfileTime = time;
  }

  if (nrs->checkpointStep || nrs->lastStep) {
    writeBudgetProfiles(time);
  }
}

struct TemperatureStatistics
{
  dfloat mean;
  dfloat rms;
  dfloat bulk;
  dfloat wall;
  dfloat wallToBulk;
  dfloat stanton;
  dfloat nusseltHalfGap;
  dfloat nusseltFullGap;
};

TemperatureStatistics temperatureStatistics(const dfloat bulkVelocity)
{
  auto mesh = nrs->scalar->mesh("temperature");
  const auto offset = nrs->fluid->fieldOffset;
  auto o_temperature = nrs->scalar->o_solution("temperature");
  auto o_rhoCp = nrs->scalar->o_transportCoeff("temperature");
  auto o_streamwiseVelocity =
      nrs->fluid->o_U.slice(0*offset, mesh->Nlocal);

  const dfloat temperatureMean = platform->linAlg->innerProd(
      mesh->Nlocal,
      mesh->o_LMM,
      o_temperature,
      platform->comm.mpiComm())/mesh->volume;

  auto o_temperatureFluctuation =
      platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  auto o_temperatureVariance =
      platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  o_temperatureFluctuation.copyFrom(o_temperature, mesh->Nlocal);
  platform->linAlg->add(mesh->Nlocal,
                        -temperatureMean,
                        o_temperatureFluctuation);
  o_temperatureVariance.copyFrom(o_temperatureFluctuation, mesh->Nlocal);
  platform->linAlg->axmy(mesh->Nlocal,
                         1.0,
                         o_temperatureFluctuation,
                         o_temperatureVariance);
  const dfloat temperatureRms = sqrt(platform->linAlg->innerProd(
      mesh->Nlocal,
      mesh->o_LMM,
      o_temperatureVariance,
      platform->comm.mpiComm())/mesh->volume);

  // Mixing-cup temperature. The rho*cp weighting is needed when the two
  // phases have different volumetric heat capacities.
  auto o_energyFlux =
      platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  auto o_temperatureEnergyFlux =
      platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  o_energyFlux.copyFrom(o_streamwiseVelocity, mesh->Nlocal);
  platform->linAlg->axmy(mesh->Nlocal, 1.0, o_rhoCp, o_energyFlux);
  o_temperatureEnergyFlux.copyFrom(o_energyFlux, mesh->Nlocal);
  platform->linAlg->axmy(mesh->Nlocal,
                         1.0,
                         o_temperature,
                         o_temperatureEnergyFlux);
  const dfloat integratedEnergyFlux = platform->linAlg->innerProd(
      mesh->Nlocal,
      mesh->o_LMM,
      o_energyFlux,
      platform->comm.mpiComm());
  const dfloat integratedTemperatureEnergyFlux =
      platform->linAlg->innerProd(mesh->Nlocal,
                                  mesh->o_LMM,
                                  o_temperatureEnergyFlux,
                                  platform->comm.mpiComm());
  const dfloat bulkTemperature = fabs(integratedEnergyFlux) > 1.0e-14
      ? integratedTemperatureEnergyFlux/integratedEnergyFlux
      : temperatureMean;

  // The temperature and velocity wall IDs describe the same two channel
  // walls. Using the velocity map avoids depending on the UDF-Neumann type ID.
  std::vector<int> wallBoundaryIds;
  for (auto &[key, bcID] : platform->app->bc->bIdToTypeId()) {
    if (key.first == "fluid velocity" &&
        bcID == bdryBase::bcType_zeroDirichlet) {
      wallBoundaryIds.push_back(key.second + 1);
    }
  }

  dfloat wallTemperature = temperatureMean;
  if (!wallBoundaryIds.empty()) {
    auto o_wallBoundaryIds = platform->device.malloc<int>(
        wallBoundaryIds.size(), wallBoundaryIds.data());
    auto o_one = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
    platform->linAlg->fill(mesh->Nlocal, 1.0, o_one);
    const dfloat wallArea =
        mesh->surfaceAreaMultiplyIntegrate(o_wallBoundaryIds, o_one);
    auto o_wallTemperature = o_temperature.slice(0, mesh->Nlocal);
    const dfloat integratedWallTemperature =
        mesh->surfaceAreaMultiplyIntegrate(o_wallBoundaryIds, o_wallTemperature);
    if (wallArea > 0.0) {
      wallTemperature = integratedWallTemperature/wallArea;
    }
  }

  const dfloat wallToBulk = wallTemperature - bulkTemperature;
  dfloat stanton = 0.0;
  dfloat nusseltHalfGap = 0.0;
  dfloat nusseltFullGap = 0.0;
  if (wallToBulk > 1.0e-14 && fabs(bulkVelocity) > 1.0e-14) {
    // Unit wall heat flux and T scale q_w/(rho_l*cp_l*Ub) give
    // St = 1/[Ub*(Tw-Tb)]. With Re based on the reference velocity,
    // Re_b = Re*Ub and Nu_H = Re_b*Pr*St.
    stanton = 1.0/(bulkVelocity*wallToBulk);
    nusseltFullGap = Re*bulkVelocity*Pr*stanton;
    nusseltHalfGap = 0.5*nusseltFullGap;
  }

  return {temperatureMean,
          temperatureRms,
          bulkTemperature,
          wallTemperature,
          wallToBulk,
          stanton,
          nusseltHalfGap,
          nusseltFullGap};
}

void monitorFlow(const dfloat time, const int tstep)
{
  static bool firstDiagnostic = true;
  if (!firstDiagnostic && tstep % 100 != 0) {
    return;
  }

  auto mesh = nrs->meshV;
  const auto offset = nrs->fluid->fieldOffset;
  auto o_U = nrs->fluid->o_U;

  const auto wallStatistics = y_p(time);
  const dfloat measuredReTau =
      wallStatistics.frictionVelocityAverage*(Re/2.0);
  const dfloat pressureGradient = activePressureGradient();

  std::array<dfloat, 3> bulkVelocity = {0.0, 0.0, 0.0};
  for (int component = 0; component < mesh->dim; ++component) {
    bulkVelocity[component] = platform->linAlg->innerProd(
                              mesh->Nlocal,
                              mesh->o_LMM,
                              o_U.slice(component*offset, mesh->Nlocal),
                              platform->comm.mpiComm())/mesh->volume;
  }

  poolDeviceMemory<dfloat> o_uPrime(nrs->fluid->fieldOffsetSum);
  poolDeviceMemory<dfloat> o_specificEnergy(offset);
  o_uPrime.copyFrom(o_U, nrs->fluid->fieldOffsetSum);
  for (int component = 0; component < mesh->dim; ++component) {
    platform->linAlg->add(mesh->Nlocal,
                          -bulkVelocity[component],
                          o_uPrime.slice(component*offset, mesh->Nlocal));
  }
  platform->linAlg->magSqrVector(mesh->Nlocal,
                                 offset,
                                 o_uPrime,
                                 o_specificEnergy);
  platform->linAlg->scale(mesh->Nlocal, 0.5, o_specificEnergy);
  const dfloat kMonitor =
      platform->linAlg->innerProd(mesh->Nlocal,
                                  mesh->o_LMM,
                                  o_specificEnergy,
                                  platform->comm.mpiComm())/mesh->volume;
  const auto thermal = temperatureStatistics(bulkVelocity[0]);

  if (platform->comm.mpiRank() == 0) {
    printf("CHANNEL mode=%s forcing=%s  t=% .6e  Ub=% .6e  utau=% .6e  "
           "ReTau=% .6e  dpdx=% .6e  kMonitor=% .6e\n",
           twoPhaseMonitorsEnabled ? "TP" : "FROZEN",
           platform->options.compareArgs("CONSTANT FLOW RATE", "TRUE")
               ? "CFR" : "DPDX",
           time,
           bulkVelocity[0],
           wallStatistics.frictionVelocityAverage,
           measuredReTau,
           pressureGradient,
           kMonitor);
    printf("THERMAL t=% .6e  Tmean=% .6e  Trms=% .6e  "
           "Tb=% .6e  Tw=% .6e  Tw-Tb=% .6e  St=% .6e  "
           "Nu_delta=% .6e  Nu_H=% .6e\n",
           time,
           thermal.mean,
           thermal.rms,
           thermal.bulk,
           thermal.wall,
           thermal.wallToBulk,
           thermal.stanton,
           thermal.nusseltHalfGap,
           thermal.nusseltFullGap);
  }

  if (twoPhaseMonitorsEnabled) {
    const dfloat gasVolume = lsDiagnostic::enclosedVolume();
    const dfloat alphaGas = gasVolume/mesh->volume;
    const dfloat relativeVolumeError =
        referenceGasVolume > 0.0
            ? fabs((gasVolume - referenceGasVolume)/referenceGasVolume)
            : 0.0;
    const dfloat interfaceArea = lsDiagnostic::interfacialArea();
    const dfloat interfaceThickness = lsDiagnostic::interfaceThickness();
    const dfloat d32 = lsDiagnostic::sauterMeanDiameter();
    const dfloat surfaceEnergy = interfaceArea/(We*mesh->volume);

    const auto liquidVelocity = lsDiagnostic::bulkPhaseAvgVelocity();
    const auto gasVelocity = lsDiagnostic::dispersedPhaseAvgVelocity();
    const std::array<dfloat, 3> slipVelocity = {
        gasVelocity[0] - liquidVelocity[0],
        gasVelocity[1] - liquidVelocity[1],
        gasVelocity[2] - liquidVelocity[2]};
    const dfloat slipMagnitude =
        sqrt(slipVelocity[0]*slipVelocity[0]
             + slipVelocity[1]*slipVelocity[1]
             + slipVelocity[2]*slipVelocity[2]);

    o_uPrime.copyFrom(o_U, nrs->fluid->fieldOffsetSum);
    for (int component = 0; component < mesh->dim; ++component) {
      platform->linAlg->add(
          mesh->Nlocal,
          -liquidVelocity[component],
          o_uPrime.slice(component*offset, mesh->Nlocal));
    }
    platform->linAlg->magSqrVector(mesh->Nlocal,
                                   offset,
                                   o_uPrime,
                                   o_specificEnergy);
    platform->linAlg->scale(mesh->Nlocal, 0.5, o_specificEnergy);
    const dfloat liquidTke = lsDiagnostic::bulkPhaseAverage(o_specificEnergy);

    o_uPrime.copyFrom(o_U, nrs->fluid->fieldOffsetSum);
    for (int component = 0; component < mesh->dim; ++component) {
      platform->linAlg->add(
          mesh->Nlocal,
          -gasVelocity[component],
          o_uPrime.slice(component*offset, mesh->Nlocal));
    }
    platform->linAlg->magSqrVector(mesh->Nlocal,
                                   offset,
                                   o_uPrime,
                                   o_specificEnergy);
    platform->linAlg->scale(mesh->Nlocal, 0.5, o_specificEnergy);
    const dfloat gasTke =
        lsDiagnostic::dispersedPhaseAverage(o_specificEnergy);

    if (platform->comm.mpiRank() == 0) {
      printf("BUBBLY t=% .6e  alphaG=% .6e  dV/V0=% .6e  "
             "area=% .6e  d32=% .6e  kL=% .6e  kG=% .6e  "
             "uL=(% .6e,% .6e,% .6e)  "
             "uG=(% .6e,% .6e,% .6e)  |slip|=% .6e\n",
             time,
             alphaGas,
             relativeVolumeError,
             interfaceArea,
             d32,
             liquidTke,
             gasTke,
             liquidVelocity[0],
             liquidVelocity[1],
             liquidVelocity[2],
             gasVelocity[0],
             gasVelocity[1],
             gasVelocity[2],
             slipMagnitude);

      fprintf(bubblyFile,
              "%.16e %.16e %.16e %.16e %.16e "
              "%.16e %.16e %.16e %.16e %.16e "
              "%.16e %.16e %.16e %.16e %.16e "
              "%.16e %.16e %.16e %.16e %.16e\n",
              time,
              alphaGas,
              gasVolume,
              relativeVolumeError,
              interfaceArea,
              interfaceThickness,
              d32,
              surfaceEnergy,
              liquidTke,
              gasTke,
              liquidVelocity[0],
              liquidVelocity[1],
              liquidVelocity[2],
              gasVelocity[0],
              gasVelocity[1],
              gasVelocity[2],
              slipVelocity[0],
              slipVelocity[1],
              slipVelocity[2],
              slipMagnitude);
      fflush(bubblyFile);
    }
  }

  firstDiagnostic = false;
}

void postProcessingPreLS(const dfloat time)
{
  collectBudgetProfiles(time);
}

void postProcessingPostLS(const dfloat time, const int tstep)
{
  monitorFlow(time, tstep);
}

#endif // POST_PROCESSING_INC
