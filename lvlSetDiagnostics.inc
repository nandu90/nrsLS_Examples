#ifndef LVLSET_DIAGNOSTICS_INC
#define LVLSET_DIAGNOSTICS_INC

#include <array>
#include <cmath>

namespace lsDiagnostic
{

// Return the diffuse-interface area
//
//                    A = integral_Omega delta dV,
//
// where delta is the regularized delta function constructed by the level-set
// plugin from CLS:
//
//       delta = [1 - (2*cls - 1)^2] / (4*interfaceWidth).
//
// CLS is generated from the transported signed-distance field, so integrating
// delta over the domain gives the diffuse-interface approximation of area.
// This routine is collective over platform->comm.mpiComm().
inline dfloat interfacialArea()
{
  auto mesh = nrs->scalar->meshV;
  const auto& o_delta = lvlSet::getDeltaFunction();

  auto interfaceArea = platform->linAlg->innerProd(mesh->Nlocal,
                       mesh->o_LMM,
                       o_delta,
                       platform->comm.mpiComm());

  return interfaceArea;
}

// Return the mean interface thickness defined by Eq. 87:
//
//       l_avg = integral_Omega H(cls) dV / integral_Omega delta(cls) dV,
//
// where H(cls) is one for 0.05 <= cls <= 0.95 and zero otherwise.
// The denominator is the diffuse interfacial area.
inline dfloat interfaceThickness()
{
  auto mesh = nrs->scalar->meshV;
  auto o_cls = nrs->scalar->o_solution("cls");

  auto o_indicator =
      platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  lvlSetInterfaceIndicator(mesh->Nlocal,
                           0.05,
                           0.95,
                           o_cls,
                           o_indicator);

  const dfloat interfaceVolume =
      platform->linAlg->innerProd(mesh->Nlocal,
                                  mesh->o_LMM,
                                  o_indicator,
                                  platform->comm.mpiComm());

  const dfloat interfaceArea = interfacialArea();
  return interfaceArea > 0.0 ? interfaceVolume / interfaceArea : 0.0;
}

// Return the volume occupied by the dispersed phase. Following the far-field
// detection in lvlSet.cpp, the smaller of the cls=0 and cls=1 phase volumes is
// identified as the dispersed (enclosed) phase.
inline dfloat enclosedVolume()
{
  static int dispersedPhase = -1;

  if (dispersedPhase < 0) {
    const dfloat phase0Volume = lvlSet::getEnclosedVol(0);
    const dfloat phase1Volume = lvlSet::getEnclosedVol(1);
    dispersedPhase = phase0Volume <= phase1Volume ? 0 : 1;
    return dispersedPhase == 0 ? phase0Volume : phase1Volume;
  }

  return lvlSet::getEnclosedVol(dispersedPhase);
}

// Return the volume occupied by the continuous bulk phase.
inline dfloat bulkVolume()
{
  static int bulkPhase = -1;

  if (bulkPhase < 0) {
    const dfloat phase0Volume = lvlSet::getEnclosedVol(0);
    const dfloat phase1Volume = lvlSet::getEnclosedVol(1);
    bulkPhase = phase0Volume > phase1Volume ? 0 : 1;
    return bulkPhase == 0 ? phase0Volume : phase1Volume;
  }

  return lvlSet::getEnclosedVol(bulkPhase);
}

// Return the integral of a scalar field over the dispersed phase:
//
//       I_d = integral_Omega alpha_d f dV.
inline dfloat dispersedPhaseIntegral(const occa::memory& o_field)
{
  auto mesh = nrs->scalar->meshV;
  static int dispersedPhase = -1;

  if (dispersedPhase < 0) {
    const dfloat phase0Volume = lvlSet::getEnclosedVol(0);
    const dfloat phase1Volume = lvlSet::getEnclosedVol(1);
    dispersedPhase = phase0Volume <= phase1Volume ? 0 : 1;
  }

  auto o_weightedField =
      platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  lvlSetPhaseWeightedField(mesh->Nlocal,
                           dispersedPhase,
                           nrs->scalar->o_solution("cls"),
                           o_field,
                           o_weightedField);

  return platform->linAlg->innerProd(mesh->Nlocal,
                                     mesh->o_LMM,
                                     o_weightedField,
                                     platform->comm.mpiComm());
}

// Return the integral of a scalar field over the continuous bulk phase:
//
//       I_b = integral_Omega alpha_b f dV.
inline dfloat bulkPhaseIntegral(const occa::memory& o_field)
{
  auto mesh = nrs->scalar->meshV;
  static int bulkPhase = -1;

  if (bulkPhase < 0) {
    const dfloat phase0Volume = lvlSet::getEnclosedVol(0);
    const dfloat phase1Volume = lvlSet::getEnclosedVol(1);
    bulkPhase = phase0Volume > phase1Volume ? 0 : 1;
  }

  auto o_weightedField =
      platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  lvlSetPhaseWeightedField(mesh->Nlocal,
                           bulkPhase,
                           nrs->scalar->o_solution("cls"),
                           o_field,
                           o_weightedField);

  return platform->linAlg->innerProd(mesh->Nlocal,
                                     mesh->o_LMM,
                                     o_weightedField,
                                     platform->comm.mpiComm());
}

// Return the volume average of a scalar field in the dispersed phase:
//
//       <f>_d = I_d / V_d.
inline dfloat dispersedPhaseAverage(const occa::memory& o_field)
{
  const dfloat volume = enclosedVolume();
  return volume > 0.0 ? dispersedPhaseIntegral(o_field) / volume : 0.0;
}

// Return the volume average of a scalar field in the continuous bulk phase:
//
//       <f>_b = I_b / V_b.
inline dfloat bulkPhaseAverage(const occa::memory& o_field)
{
  const dfloat volume = bulkVolume();
  return volume > 0.0 ? bulkPhaseIntegral(o_field) / volume : 0.0;
}

// Return the dispersed-phase centroid.
inline std::array<dfloat, 3> dispersedPhaseCentroid()
{
  auto mesh = nrs->scalar->meshV;
  return {dispersedPhaseAverage(mesh->o_x),
          dispersedPhaseAverage(mesh->o_y),
          mesh->dim == 3 ? dispersedPhaseAverage(mesh->o_z) : 0.0};
}

// Return the dispersed-phase mean velocity.
inline std::array<dfloat, 3> dispersedPhaseAvgVelocity()
{
  const dlong fieldOffset = nrs->fluid->fieldOffset;
  auto o_U = nrs->fluid->o_U;

  return {dispersedPhaseAverage(o_U.slice(0 * fieldOffset, fieldOffset)),
          dispersedPhaseAverage(o_U.slice(1 * fieldOffset, fieldOffset)),
          nrs->meshV->dim == 3
              ? dispersedPhaseAverage(o_U.slice(2 * fieldOffset, fieldOffset))
              : 0.0};
}

// Return the continuous bulk-phase mean velocity.
inline std::array<dfloat, 3> bulkPhaseAvgVelocity()
{
  const dlong fieldOffset = nrs->fluid->fieldOffset;
  auto o_U = nrs->fluid->o_U;

  return {bulkPhaseAverage(o_U.slice(0 * fieldOffset, fieldOffset)),
          bulkPhaseAverage(o_U.slice(1 * fieldOffset, fieldOffset)),
          nrs->meshV->dim == 3
              ? bulkPhaseAverage(o_U.slice(2 * fieldOffset, fieldOffset))
              : 0.0};
}

// Return the dispersed-minus-bulk mean pressure jump.
inline dfloat meanPressureJump()
{
  return dispersedPhaseAverage(nrs->fluid->o_P) -
         bulkPhaseAverage(nrs->fluid->o_P);
}

// Return the diameter of a circle/sphere with the dispersed-phase volume.
inline dfloat equivalentDiameter()
{
  const dfloat pi = std::acos(-1.0);
  const dfloat volume = enclosedVolume();

  if (nrs->scalar->meshV->dim == 2) {
    return 2.0 * std::sqrt(volume / pi);
  }
  return std::cbrt(6.0 * volume / pi);
}

// Return the Sauter mean diameter based on dispersed volume and interface area.
inline dfloat sauterMeanDiameter()
{
  const dfloat area = interfacialArea();
  if (area <= 0.0) {
    return 0.0;
  }

  const dfloat coefficient = nrs->scalar->meshV->dim == 2 ? 4.0 : 6.0;
  return coefficient * enclosedVolume() / area;
}

// Return 2-D circularity or 3-D sphericity; either is one for a circle/sphere.
inline dfloat sphericity()
{
  const dfloat pi = std::acos(-1.0);
  const dfloat volume = enclosedVolume();
  const dfloat area = interfacialArea();

  if (area <= 0.0) {
    return 0.0;
  }
  if (nrs->scalar->meshV->dim == 2) {
    return 2.0 * std::sqrt(pi * volume) / area;
  }
  return std::cbrt(pi) * std::pow(6.0 * volume, 2.0 / 3.0) / area;
}

// Return the delta-weighted interface integral of a scalar field.
inline dfloat interfaceIntegral(const occa::memory& o_field)
{
  auto mesh = nrs->scalar->meshV;
  const auto& o_delta = lvlSet::getDeltaFunction();
  auto o_weightedField =
      platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);

  lvlSetMultiplyFields(mesh->Nlocal,
                       o_delta,
                       o_field,
                       o_weightedField);

  return platform->linAlg->innerProd(mesh->Nlocal,
                                     mesh->o_LMM,
                                     o_weightedField,
                                     platform->comm.mpiComm());
}

// Return the delta-weighted mean interface curvature.
inline dfloat meanCurvature()
{
  auto mesh = nrs->scalar->meshV;
  auto o_normals =
      platform->deviceMemoryPool.reserve<dfloat>(3 * nrs->scalar->vFieldOffset);

  const bool averageNormals =
      platform->options.compareArgs("LVLSET NORMAL AVERAGING", "TRUE");
  lvlSet::normalVector(nrs->scalar->o_solution("tls"),
                       o_normals,
                       averageNormals);
  const auto& o_curvature = lvlSet::getCurvature(o_normals);

  const dfloat area = interfacialArea();
  return area > 0.0 ? interfaceIntegral(o_curvature) / area : 0.0;
}

// Return the delta-weighted mean velocity at the interface.
inline std::array<dfloat, 3> interfaceVelocity()
{
  const dfloat area = interfacialArea();
  if (area <= 0.0) {
    return {0.0, 0.0, 0.0};
  }

  const dlong fieldOffset = nrs->fluid->fieldOffset;
  auto o_U = nrs->fluid->o_U;
  return {interfaceIntegral(o_U.slice(0 * fieldOffset, fieldOffset)) / area,
          interfaceIntegral(o_U.slice(1 * fieldOffset, fieldOffset)) / area,
          nrs->meshV->dim == 3
              ? interfaceIntegral(o_U.slice(2 * fieldOffset, fieldOffset)) / area
              : 0.0};
}

} // namespace lsDiagnostic

#endif
