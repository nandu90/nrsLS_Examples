from cycler import cycler
import glob
from mpi4py import MPI
import numpy as np
import os
import time
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt


def plotnow(fname, xlabel, ylabel, x, y, labels, ptype="line",
            linestyles=[], markers=[], ylim=[], xlim=[]):
    default_cycler = (
        cycler(color=["#0072B2", "#D55E00", "#009E73",
                      "#CC0000", "#990099"])*
        cycler(linestyle=["-"])*cycler(marker=[""])
    )
    plt.rc("lines", linewidth=1)
    plt.rc("axes", prop_cycle=default_cycler)
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)

    if ylim != []:
        ax.set_ylim(ylim[0], ylim[1])

    if xlim != []:
        ax.set_xlim(xlim[0], xlim[1])

    for i in range(len(y)):
        if ptype == "line":
            ax.plot(x[i], y[i], label=labels[i],
                    linestyle=linestyles[i], marker=markers[i],
                    linewidth=2.0)
        elif ptype == "semilogx":
            ax.semilogx(x[i], y[i], label=labels[i],
                        linestyle=linestyles[i], marker=markers[i],
                        linewidth=2.0)
        elif ptype == "semilogy":
            ax.semilogy(x[i], y[i], label=labels[i],
                        linestyle=linestyles[i], marker=markers[i],
                        linewidth=2.0)
        else:
            ax.loglog(x[i], y[i], label=labels[i],
                      linestyle=linestyles[i], marker=markers[i],
                      linewidth=2.0)

    ax.grid()
    ax.legend(loc="best", fontsize=12)
    fig.savefig(fname + ".pdf", bbox_inches="tight", dpi=100)
    fig.savefig(fname + ".png", bbox_inches="tight", dpi=100)
    plt.close()

    return


def fourierSpectrum(u, v, w, dx):
    nx, ny, nz = u.shape
    npts = nx*ny*nz

    uHat = np.fft.fftn(u)/npts
    vHat = np.fft.fftn(v)/npts
    wHat = np.fft.fftn(w)/npts

    kx = 2.0*np.pi*np.fft.fftfreq(nx, d=dx)
    ky = 2.0*np.pi*np.fft.fftfreq(ny, d=dx)
    kz = 2.0*np.pi*np.fft.fftfreq(nz, d=dx)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")

    kMagnitude = np.sqrt(kx*kx + ky*ky + kz*kz)
    kShell = np.floor(kMagnitude + 0.5).astype(int)

    modalEnergy = 0.5*(np.abs(uHat)**2 +
                       np.abs(vHat)**2 +
                       np.abs(wHat)**2)

    spectrum = np.bincount(kShell.ravel(),
                           weights=modalEnergy.ravel())
    k = np.arange(spectrum.size)

    return k, spectrum


def gaussianFilter(field, kMagnitude, filterWidth):
    sigma = filterWidth/(2.0*np.sqrt(2.0*np.log(2.0)))
    transferFunction = np.exp(-0.5*sigma*sigma*kMagnitude*kMagnitude)

    return np.fft.ifftn(np.fft.fftn(field)*transferFunction).real


def filteredPhaseEnergy(u, v, w, phiCarrier,
                        rhoCarrier, rhoDispersed, kFilter, dx):
    nx, ny, nz = u.shape

    kx = 2.0*np.pi*np.fft.fftfreq(nx, d=dx)
    ky = 2.0*np.pi*np.fft.fftfreq(ny, d=dx)
    kz = 2.0*np.pi*np.fft.fftfreq(nz, d=dx)
    kx, ky, kz = np.meshgrid(kx, ky, kz, indexing="ij")
    kMagnitude = np.sqrt(kx*kx + ky*ky + kz*kz)

    phiCarrier = np.clip(phiCarrier, 0.0, 1.0)
    phiDispersed = 1.0 - phiCarrier

    carrierEnergy = np.zeros(kFilter.size)
    dispersedEnergy = np.zeros(kFilter.size)
    interactionEnergy = np.zeros(kFilter.size)

    for i, kCutoff in enumerate(kFilter):
        filterWidth = 2.0*np.pi/kCutoff

        carrierMass = gaussianFilter(
            rhoCarrier*phiCarrier, kMagnitude, filterWidth)
        dispersedMass = gaussianFilter(
            rhoDispersed*phiDispersed, kMagnitude, filterWidth)

        carrierMomentum = []
        dispersedMomentum = []

        for velocity in [u, v, w]:
            carrierMomentum.append(gaussianFilter(
                rhoCarrier*phiCarrier*velocity,
                kMagnitude, filterWidth))
            dispersedMomentum.append(gaussianFilter(
                rhoDispersed*phiDispersed*velocity,
                kMagnitude, filterWidth))

        carrierResolved = np.zeros_like(u)
        dispersedResolved = np.zeros_like(u)
        relativeVelocitySquared = np.zeros_like(u)

        for carrier, dispersed in zip(carrierMomentum,
                                      dispersedMomentum):
            carrierResolved += np.divide(
                carrier*carrier,
                carrierMass,
                out=np.zeros_like(carrier),
                where=carrierMass > 1.0e-14)
            dispersedResolved += np.divide(
                dispersed*dispersed,
                dispersedMass,
                out=np.zeros_like(dispersed),
                where=dispersedMass > 1.0e-14)

            carrierVelocity = np.divide(
                carrier,
                carrierMass,
                out=np.zeros_like(carrier),
                where=carrierMass > 1.0e-14)
            dispersedVelocity = np.divide(
                dispersed,
                dispersedMass,
                out=np.zeros_like(dispersed),
                where=dispersedMass > 1.0e-14)
            relativeVelocitySquared += (
                carrierVelocity - dispersedVelocity)**2

        mixtureMass = carrierMass + dispersedMass
        interactionResolved = np.divide(
            carrierMass*dispersedMass,
            mixtureMass,
            out=np.zeros_like(mixtureMass),
            where=mixtureMass > 1.0e-14)
        interactionResolved *= relativeVelocitySquared

        carrierEnergy[i] = 0.5*np.mean(carrierResolved)
        dispersedEnergy[i] = 0.5*np.mean(dispersedResolved)
        interactionEnergy[i] = 0.5*np.mean(interactionResolved)

    carrierSpectrum = np.gradient(carrierEnergy, kFilter)
    dispersedSpectrum = np.gradient(dispersedEnergy, kFilter)
    interactionSpectrum = np.gradient(interactionEnergy, kFilter)

    return carrierSpectrum, dispersedSpectrum, interactionSpectrum


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    inputFiles = sorted(glob.glob("interpolatedData/*.npz"))
    myInputFiles = inputFiles[rank::size]
    outputDir = "spectraData"

    rhoCarrier = 1.0
    rhoDispersed = 0.1
    phaseTolerance = 1.0e-8

    if len(inputFiles) == 0:
        raise RuntimeError("No files found in interpolatedData")

    if rank == 0:
        os.makedirs(outputDir, exist_ok=True)

    comm.Barrier()

    with np.load(inputFiles[0]) as firstData:
        n = firstData["x"].size

    numberOfFilters = 64
    kFilter = np.logspace(np.log10(0.5),
                          np.log10(n/2.0), numberOfFilters)

    fourierSpectra = []
    carrierSpectra = []
    dispersedSpectra = []
    interactionSpectra = []
    times = []
    carrierVolumeFractions = []
    dispersedVolumeFractions = []

    k = None

    for inputFile in myInputFiles:
        data = np.load(inputFile)

        x = data["x"]
        u = data["u"]
        v = data["v"]
        w = data["w"]
        phiCarrier = np.clip(data["cls"], 0.0, 1.0)

        if np.mean(1.0 - phiCarrier) < phaseTolerance:
            phiCarrier = np.ones_like(phiCarrier)

        if not (u.shape[0] == u.shape[1] == u.shape[2]):
            raise RuntimeError("The uniform grid must be cubic")

        dx = x[1] - x[0]
        n = u.shape[0]

        u = u - np.mean(u)
        v = v - np.mean(v)
        w = w - np.mean(w)

        k, spectrum = fourierSpectrum(u, v, w, dx)

        alphaCarrier = np.mean(phiCarrier)
        alphaDispersed = 1.0 - alphaCarrier

        if alphaDispersed < phaseTolerance:
            carrierSpectrum = np.zeros_like(kFilter)
            dispersedSpectrum = np.zeros_like(kFilter)
            interactionSpectrum = np.zeros_like(kFilter)
        else:
            carrierSpectrum, dispersedSpectrum, interactionSpectrum = \
                filteredPhaseEnergy(
                    u, v, w, phiCarrier,
                    rhoCarrier, rhoDispersed, kFilter, dx)

        kineticEnergy = 0.5*np.mean(u*u + v*v + w*w)
        spectralEnergy = np.sum(spectrum)

        print("rank=%d  %s  t=%g  k=%g  sum(E)=%g  error=%g" %
              (rank, inputFile, float(data["time"]), kineticEnergy,
               spectralEnergy, abs(kineticEnergy - spectralEnergy)))

        times.append(float(data["time"]))
        fourierSpectra.append(spectrum)
        carrierSpectra.append(carrierSpectrum)
        dispersedSpectra.append(dispersedSpectrum)
        interactionSpectra.append(interactionSpectrum)
        carrierVolumeFractions.append(alphaCarrier)
        dispersedVolumeFractions.append(alphaDispersed)

    localResults = {
        "time": np.array(times),
        "k": k,
        "fourier": np.array(fourierSpectra),
        "carrier": np.array(carrierSpectra),
        "dispersed": np.array(dispersedSpectra),
        "interaction": np.array(interactionSpectra),
        "alphaCarrier": np.array(carrierVolumeFractions),
        "alphaDispersed": np.array(dispersedVolumeFractions),
    }
    gatheredResults = comm.gather(localResults, root=0)

    if rank != 0:
        return

    nonemptyResults = [result for result in gatheredResults
                       if result["time"].size > 0]
    k = nonemptyResults[0]["k"]

    times = np.concatenate([result["time"]
                            for result in nonemptyResults])
    fourierSpectra = np.concatenate([result["fourier"]
                                     for result in nonemptyResults], axis=0)
    carrierSpectra = np.concatenate([result["carrier"]
                                     for result in nonemptyResults], axis=0)
    dispersedSpectra = np.concatenate([result["dispersed"]
                                       for result in nonemptyResults], axis=0)
    interactionSpectra = np.concatenate([result["interaction"]
                                         for result in nonemptyResults], axis=0)
    carrierVolumeFractions = np.concatenate([
        result["alphaCarrier"] for result in nonemptyResults])
    dispersedVolumeFractions = np.concatenate([
        result["alphaDispersed"] for result in nonemptyResults])

    timeOrder = np.argsort(times)
    times = times[timeOrder]
    fourierSpectra = fourierSpectra[timeOrder]
    carrierSpectra = carrierSpectra[timeOrder]
    dispersedSpectra = dispersedSpectra[timeOrder]
    interactionSpectra = interactionSpectra[timeOrder]
    carrierVolumeFractions = carrierVolumeFractions[timeOrder]
    dispersedVolumeFractions = dispersedVolumeFractions[timeOrder]

    alphaCarrier = np.mean(carrierVolumeFractions)
    alphaDispersed = np.mean(dispersedVolumeFractions)
    singlePhase = alphaDispersed < phaseTolerance

    if singlePhase:
        carrierMean = np.zeros_like(kFilter)
        dispersedMean = np.zeros_like(kFilter)
        interactionMean = np.zeros_like(kFilter)
        carrierMeanSpecific = np.zeros_like(kFilter)
        dispersedMeanSpecific = np.zeros_like(dispersedMean)
    else:
        carrierMean = np.mean(carrierSpectra, axis=0)
        dispersedMean = np.mean(dispersedSpectra, axis=0)
        interactionMean = np.mean(interactionSpectra, axis=0)
        carrierMeanSpecific = carrierMean/alphaCarrier
        dispersedMeanSpecific = dispersedMean/alphaDispersed

    np.savez(
        os.path.join(outputDir, "spectra.npz"),
        time=np.array(times),
        k=k,
        fourier=fourierSpectra,
        fourierMean=np.mean(fourierSpectra, axis=0),
        kFilter=kFilter,
        carrier=carrierSpectra,
        dispersed=dispersedSpectra,
        interaction=interactionSpectra,
        carrierMean=carrierMean,
        dispersedMean=dispersedMean,
        interactionMean=interactionMean,
        carrierMeanSpecific=carrierMeanSpecific,
        dispersedMeanSpecific=dispersedMeanSpecific,
        alphaCarrier=np.array(alphaCarrier),
        alphaDispersed=np.array(alphaDispersed),
        singlePhase=np.array(singlePhase),
    )

    fourierOutput = np.column_stack((
        k,
        np.mean(fourierSpectra, axis=0),
        np.std(fourierSpectra, axis=0),
    ))
    np.savetxt(
        os.path.join(outputDir, "fourierSpectrum.dat"),
        fourierOutput,
        header="k Emean Estd",
    )

    phaseOutput = np.column_stack((
        kFilter,
        carrierMean,
        dispersedMean,
        interactionMean,
        carrierMeanSpecific,
        dispersedMeanSpecific,
    ))
    np.savetxt(
        os.path.join(outputDir, "phaseSpectrum.dat"),
        phaseOutput,
        header=("k Ecarrier Edispersed Einteraction "
                "EcarrierSpecific EdispersedSpecific"),
    )

    fourierMean = np.mean(fourierSpectra, axis=0)
    maximumWavenumber = min(128, n//2)
    plotMask = (k >= 1) & (k <= maximumWavenumber)
    referenceMask = (k >= 5) & (k <= min(40, maximumWavenumber))

    plotK = k[plotMask]
    plotSpectrum = fourierMean[plotMask]
    referenceK = k[referenceMask]

    referenceWavenumber = 10
    referenceIndex = np.argmin(np.abs(k - referenceWavenumber))
    referenceSpectrum = fourierMean[referenceIndex]*(
        referenceK/k[referenceIndex])**(-5.0/3.0)
    xPadding = 1.15

    plotnow(
        os.path.join(outputDir, "fourierSpectrum"),
        r"$\kappa$",
        r"$E(\kappa)$",
        [plotK, referenceK],
        [plotSpectrum, referenceSpectrum],
        ["Fourier spectrum", r"$\kappa^{-5/3}$"],
        linestyles=["-", "--"],
        markers=["o", ""],
        ptype="loglog",
        xlim=[1.0/xPadding, xPadding*maximumWavenumber],
    )

    if not singlePhase:
        plotnow(
            os.path.join(outputDir, "filteredPhaseSpectrum"),
            r"$\kappa_\ell$",
            r"$E_\ell/\langle\phi\rangle$",
            [kFilter, kFilter],
            [carrierMeanSpecific, dispersedMeanSpecific],
            ["Carrier phase", "Dispersed phase"],
            linestyles=["-", "--"],
            markers=["", ""],
            ptype="loglog",
        )

    print("Wrote spectra to %s" % outputDir)
    print("Single phase: %s" % singlePhase)
    print("Carrier volume fraction: %g" % alphaCarrier)
    print("Dispersed volume fraction: %g" % alphaDispersed)

    return


if __name__ == "__main__":
    starttime = time.time()
    main()

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("--- Code ran in %s seconds ---" %
              (time.time() - starttime))
