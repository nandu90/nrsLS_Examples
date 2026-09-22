static int ciMode = 0;
static dfloat initialThickness = 0.0;
static occa::memory o_constantField;

void ciSetup(MPI_Comm comm, setupAide& options)
{
  options.setArgs("CHECKPOINT INTERVAL", "0");
  options.setArgs("NUMBER TIMESTEPS", "100");
  options.setArgs("DT", to_string_f(1.0e-3));
}

void ciInitialize()
{
  auto mesh = nrs->meshV;
  o_constantField = platform->device.malloc<dfloat>(mesh->Nlocal);
  platform->linAlg->fill(mesh->Nlocal, 2.0, o_constantField);
  initialThickness = lsDiagnostic::interfaceThickness();
}

void ciTestDiagnostics(const dfloat time)
{
  const dfloat pi = std::acos(-1.0);
  const dfloat epsilon = lvlSet::getInterfaceWidth();

  // For cls = 0.5*(tanh(phi/(2*epsilon)) + 1) and
  // phi = radialDistance - r, integrating 1-cls over a 2-D circle gives the
  // sharp area plus the diffuse-profile correction below.
  const dfloat exactDispersedVolume = pi * r * r + pi * pi * pi * epsilon * epsilon / 3.0;
  const dfloat exactBulkVolume = 1.0 - exactDispersedVolume;
  const dfloat exactArea = 2.0 * pi * r;
  const dfloat expectedCenterX = xc + ux * time;
  const dfloat expectedCenterY = yc + uy * time;

  const dfloat area = lsDiagnostic::interfacialArea();
  const dfloat thickness = lsDiagnostic::interfaceThickness();
  const dfloat dispersedVolume = lsDiagnostic::enclosedVolume();
  const dfloat continuousVolume = lsDiagnostic::bulkVolume();
  const dfloat dispersedIntegral = lsDiagnostic::dispersedPhaseIntegral(o_constantField);
  const dfloat bulkIntegral = lsDiagnostic::bulkPhaseIntegral(o_constantField);
  const dfloat dispersedAverage = lsDiagnostic::dispersedPhaseAverage(o_constantField);
  const dfloat bulkAverage = lsDiagnostic::bulkPhaseAverage(o_constantField);
  const auto centroid = lsDiagnostic::dispersedPhaseCentroid();
  const auto dispersedVelocity = lsDiagnostic::dispersedPhaseAvgVelocity();
  const auto bulkVelocity = lsDiagnostic::bulkPhaseAvgVelocity();
  const dfloat pressureJump = lsDiagnostic::meanPressureJump();
  const dfloat diameter = lsDiagnostic::equivalentDiameter();
  const dfloat sauterDiameter = lsDiagnostic::sauterMeanDiameter();
  const dfloat shapeFactor = lsDiagnostic::sphericity();
  const dfloat interfaceIntegral = lsDiagnostic::interfaceIntegral(o_constantField);
  const dfloat curvature = lsDiagnostic::meanCurvature();
  const auto velocityAtInterface = lsDiagnostic::interfaceVelocity();

  const dfloat expectedDiameter = std::cbrt(6.0 * dispersedVolume / pi);
  const dfloat expectedSauter = 6.0 * dispersedVolume / area;
  const dfloat expectedSphericity = std::cbrt(pi) * std::pow(6.0 * dispersedVolume, 2.0 / 3.0) / area;

  const auto close = [](dfloat value, dfloat exact, dfloat tolerance)
  {
    return std::abs(value - exact) <= tolerance;
  };

  if (platform->comm.mpiRank() == 0) {
    CiEvalTest("diagnostic interfacial area: " + to_string_f(area),
               close(area, exactArea, 0.08));

    CiEvalTest("diagnostic interface thickness: " + to_string_f(thickness),
               close(thickness, initialThickness, 0.25 * initialThickness));

    CiEvalTest("diagnostic dispersed volume: " + to_string_f(dispersedVolume),
               close(dispersedVolume, exactDispersedVolume, 5.0e-4));

    CiEvalTest("diagnostic bulk volume: " + to_string_f(continuousVolume),
               close(continuousVolume, exactBulkVolume, 5.0e-4));

    CiEvalTest("diagnostic dispersed integral: " + to_string_f(dispersedIntegral),
               close(dispersedIntegral, 2.0 * dispersedVolume, 1.0e-10));

    CiEvalTest("diagnostic bulk integral: " + to_string_f(bulkIntegral),
               close(bulkIntegral, 2.0 * continuousVolume, 1.0e-10));

    CiEvalTest("diagnostic phasic averages",
               close(dispersedAverage, 2.0, 1.0e-10) &&
               close(bulkAverage, 2.0, 1.0e-10));

    CiEvalTest("diagnostic dispersed centroid",
               close(centroid[0], expectedCenterX, 1.0e-2) &&
               close(centroid[1], expectedCenterY, 1.0e-2) &&
               close(centroid[2], 0.5, 1.0e-10));

    CiEvalTest("diagnostic dispersed velocity",
               close(dispersedVelocity[0], ux, 1.0e-10) &&
               close(dispersedVelocity[1], uy, 1.0e-10) &&
               close(dispersedVelocity[2], 0.0, 1.0e-10));

    CiEvalTest("diagnostic bulk velocity",
               close(bulkVelocity[0], ux, 1.0e-10) &&
               close(bulkVelocity[1], uy, 1.0e-10) &&
               close(bulkVelocity[2], 0.0, 1.0e-10));

    CiEvalTest("diagnostic pressure jump: " + to_string_f(pressureJump),
               close(pressureJump, 0.0, 1.0e-10));

    CiEvalTest("diagnostic equivalent diameter: " + to_string_f(diameter),
               close(diameter, expectedDiameter, 1.0e-12));

    CiEvalTest("diagnostic Sauter mean diameter: " + to_string_f(sauterDiameter),
               close(sauterDiameter, expectedSauter, 1.0e-12));

    CiEvalTest("diagnostic sphericity: " + to_string_f(shapeFactor),
               close(shapeFactor, expectedSphericity, 1.0e-12));

    CiEvalTest("diagnostic interface integral: " + to_string_f(interfaceIntegral),
               close(interfaceIntegral, 2.0 * area, 1.0e-10));

    CiEvalTest("diagnostic mean curvature: " + to_string_f(curvature),
               close(std::abs(curvature), 1.0 / r, 1.0));

    CiEvalTest("diagnostic interface velocity",
               close(velocityAtInterface[0], ux, 1.0e-10) &&
               close(velocityAtInterface[1], uy, 1.0e-10) &&
               close(velocityAtInterface[2], 0.0, 1.0e-10));
  }
}
