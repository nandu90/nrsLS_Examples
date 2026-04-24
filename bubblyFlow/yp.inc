/*
  Usage:
  include yp.inc in udf file
  include yp.oudf in the oudf block in udf file
  
  Call options (from UDF_ExecuteStep):

      To only print min/max/avg values to logfile and use inbuild cheap_dist function for ywd:
      y_p(time);
      
      To also write y_p to field file:
      y_p(time, true)
      


      User specified wall distance and boundary IDs. Print only min/max/avg values to logfile
      y_p(time, nbID, o_bID, o_ywd)
      
      To also write y_p to field file:
      y_p(time, nbID, o_bID, o_ywd, true)     
 */

static deviceMemory<dfloat> o_yp;
static std::unique_ptr<iofld> ypWriter;
static int outfldCounter = 0;

void y_p(double time, const bool iffld = false);
void y_p(double time, const int &nbID, const occa::memory &o_bID, const occa::memory &o_ywd, const bool iffld = false);

dfloat reduce_sum(const std::vector<dfloat> tmp, const dlong N)
{
  dfloat sum = 0.0;
  
  for (dlong i = 0; i < N; i++) {
    sum += tmp[i];
  }

  MPI_Allreduce(MPI_IN_PLACE, &sum, 1, MPI_DFLOAT, MPI_SUM, platform->comm.mpiComm());

  return sum;
}

dfloat reduce_min(const std::vector<dfloat> tmp, const dlong N)
{
  dfloat smin = 1e9;
  
  for (dlong i = 0; i < N; i++) {
    smin = fmin(smin, tmp[i]);
  }

  MPI_Allreduce(MPI_IN_PLACE, &smin, 1, MPI_DFLOAT, MPI_MIN, platform->comm.mpiComm());

  return smin;
}

dfloat reduce_max(const std::vector<dfloat> tmp, const dlong N)
{
  dfloat smax = -1e9;
  
  for (dlong i = 0; i < N; i++) {
    smax = fmax(smax, tmp[i]);
  }

  MPI_Allreduce(MPI_IN_PLACE, &smax, 1, MPI_DFLOAT, MPI_MAX, platform->comm.mpiComm());

  return smax;
}

void y_p(double time, const bool iffld)
{
  auto mesh = nrs->meshV;
  
  std::vector<int> wbID;
  for (auto &[key, bcID] : platform->app->bc->bIdToTypeId()) {
    const auto field = key.first;
    if (field == "fluid velocity") {
      if (bcID == bdryBase::bcType_zeroDirichlet) {
	wbID.push_back(key.second + 1);
      }
    }
  }
  auto o_wbID = platform->device.malloc<int>(wbID.size(), wbID.data());
  auto o_ywd = mesh->minDistance(wbID.size(), o_wbID, "cheap_dist");

  y_p(time, wbID.size(), o_wbID, o_ywd, iffld);
}

void y_p(double time, const int &nbID, const occa::memory &o_bID, const occa::memory &o_ywd, const bool iffld)
{
  auto mesh = nrs->meshV;
  
  auto o_Sij = nrs->strainRate();
  
  auto o_mind = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements);
  auto o_maxd = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements);
  auto o_sumd = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nelements);
  
  if(iffld) o_yp.resize(nrs->meshV->Nlocal); 
  
  yplusKernel(mesh->Nelements,
	      nrs->fieldOffset,
	      nbID,
	      o_bID,
	      static_cast<int>(iffld),
	      mesh->o_sgeo,
	      mesh->o_vmapM,
	      mesh->o_EToB,
	      nrs->fluid->o_rho,
	      nrs->fluid->o_mue,
	      o_Sij,
	      o_ywd,
	      o_yp,
	      o_mind,
	      o_maxd,
	      o_sumd);

  static std::vector<dfloat> tmp;
  if (tmp.size() < o_sumd.size()) {
    tmp.resize(o_sumd.size());
  }

  auto o_tmp2 = platform->deviceMemoryPool.reserve<dfloat>(mesh->Nlocal);
  platform->linAlg->fill(mesh->Nlocal, 1.0, o_tmp2);
  const auto areaWall = mesh->surfaceAreaMultiplyIntegrate(o_bID, o_tmp2);

  o_sumd.copyTo(tmp.data());
  const dfloat sumd = reduce_sum(tmp, mesh->Nelements);

  o_mind.copyTo(tmp.data());
  const dfloat mind = reduce_min(tmp, mesh->Nelements);

  o_maxd.copyTo(tmp.data());
  const dfloat maxd = reduce_max(tmp, mesh->Nelements);

  if(platform->comm.mpiRank() == 0)
    std::cout << std::scientific << "\n      y+ min: " << mind <<" max: "<< maxd <<" avg: "<< sumd/areaWall<<"\n\n";

  if(iffld){
    if(!ypWriter) {
      ypWriter = iofldFactory::create();

      ypWriter->open(mesh, iofld::mode::write, "yplus");

      if (platform->options.compareArgs("CHECKPOINT PRECISION", "FP32")) {
        ypWriter->writeAttribute("precision", "32");
      } else {
        ypWriter->writeAttribute("precision", "64");
      }
      ypWriter->addVariable("scalar00", std::vector<deviceMemory<dfloat>>{o_yp});
    }

    ypWriter->writeAttribute("outputmesh", (!outfldCounter) ? "true" : "false");
    ypWriter->addVariable("time", const_cast<double &>(time));
    ypWriter->process();
    outfldCounter++;
  }
}
