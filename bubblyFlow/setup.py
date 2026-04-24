from cycler import cycler
import math
import numpy as np
import os
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.linalg import lstsq
from scipy.optimize import curve_fit
from scipy import stats
from scipy import optimize
import scipy.interpolate as interpolate
import re

def plotnow(fname,xlabel,ylabel,x,y,labels,ptype='line',linestyles=[],markers=[],ylim=[],xlim=[]):
    default_cycler = (cycler(color=['#0072B2','#D55E00','#009E73','#CC0000','#990099'])*\
                      cycler(linestyle=['-'])*cycler(marker=['']))
    plt.rc('lines',linewidth=1)
    plt.rc('axes',prop_cycle=default_cycler)
    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111)  

    ax.set_xlabel(xlabel,fontsize=15)
    ax.set_ylabel(ylabel,fontsize=15)
    ax.tick_params(axis='both',labelsize=12)

    if(ylim != []):
        ax.set_ylim(ylim[0],ylim[1])

    if(xlim != []):
        ax.set_xlim(xlim[0],xlim[1])

    # if(len(linestyles) == 0):
    #     linestyles = ['-']*len(x)
    #     markers = ['']*len(x)

    for i in range(len(y)):
        if(ptype=='line'):
            ax.plot(x[i],y[i],label=labels[i],linestyle=linestyles[i],marker=markers[i],linewidth=2.0)
        elif(ptype=='semilogx'):
            ax.semilogx(x[i],y[i],label=labels[i],linestyle=linestyles[i],marker=markers[i],linewidth=2.0)
        elif(ptype=='semilogy'):
            ax.semilogy(x[i],y[i],label=labels[i],linestyle=linestyles[i],marker=markers[i],linewidth=2.0)
        else:
            ax.loglog(x[i],y[i],label=labels[i],linestyle=linestyles[i],marker=markers[i],linewidth=2.0)
    

    #ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax.grid()
    ax.legend(loc='best',fontsize=12)
    fig.savefig(fname+'.pdf',\
                bbox_inches='tight',dpi=100)
    fig.savefig(fname+'.png',\
                bbox_inches='tight',dpi=100)
    plt.close()
    return

def main():
    
    Re_tau = 280
    delta = 1
    charl = 2 * delta
    charU = 1

    #first point off wall
    yplus = 1
    y = charl * yplus / Re_tau
    print("y wall = ", y)
    
    targetN = 7
    #streamwise spacing
    xL = 2 * math.pi * delta
    n_trygg = 512
    nelem = xL / (xL / n_trygg * targetN)
    print("Estimate streamwise element = ", nelem)

    #spanwise spacing
    yL = 2 * delta
    n_trygg = 192
    nelem = yL / (yL / n_trygg * targetN)
    print("Estimate spanwise element = ", nelem)

    zL = xL / 2

    #number of bubbles
    vf = 0.03
    vol = xL * yL * zL
    D = 0.3 * delta
    volb = (4.0/3.0) * math.pi * (D/2.0)**3.0
    nb = vf * vol / volb
    print("Void fraction = ",vf)
    print("Bubble Dia = ", D)
    print("NUmber of bubbles = ", nb)

    #Compute Re, We and Fr based on consistent len scales
    Re = 5600 #based on charl

    rhoRatio = 40
    rhol = 1
    rhog = rhol / rhoRatio

    muRatio = 6.07
    mul = rhol * charU * charl / Re
    mug = mul / muRatio

    nul = mul / rhol

    Eo = 0.9 #based on D
    An = 2.43e4 #Archimedes number based on D
    sigma = (An * nul**2.0 * rhol) / (Eo * D)
    g = An * nul**2.0 / D**3.0
    
    We = (rhol * charU**2 * charl) / sigma
    Fr = charU**2 / g / charl

    print("Reynolds number = ",Re)
    print("Weber number = ",We)
    print("Froude number = ",Fr)

    return

if __name__=="__main__":
    starttime = time.time()
    main()
    print('--- Code ran in %s seconds ---'%(time.time()-starttime))
