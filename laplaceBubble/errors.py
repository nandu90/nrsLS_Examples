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
    default_cycler = (cycler(color=['#0072B2',  
                                    '#D55E00',  
                                    '#009E73',  
                                    '#CC79A7',  
                                    '#6A3D9A',  
                                    '#E7298A',  
                                    '#A6761D',  
                                    '#000000'])*cycler(linestyle=['-'])*cycler(marker=['']))
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

    ax.grid(which='both', linestyle=':', linewidth=0.6, alpha=0.7)
    ax.legend(loc='best',fontsize=12)
    # ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), frameon=True)
    # fig.tight_layout(rect=[0, 0, 0.82, 1])
    # fig.savefig(fname+'.pdf',\
    #             bbox_inches='tight',dpi=100)
    fig.savefig(fname+'.png',\
                bbox_inches='tight',dpi=100)
    plt.close()
    return

def extractData(f):
    data = np.loadtxt(f)
    time = data[:,0]
    rms = data[:,2]
    divu = data[:,3]

    return time,rms,divu

def main():
    time = []
    rms = []
    divu = []

    files = ['rho1mu100.dat','rho10mu100.dat','rho100mu100.dat','rho1000mu100.dat','rho10000mu100.dat','rho10000mu100_filtered.dat']
    labels = ['$\\rho_r=10^0$','$\\rho_r=10^1$','$\\rho_r=10^2$','$\\rho_r=10^3$','$\\rho_r=10^4$','$\\rho_r=10^4,filtered$']
    linestyles = ['--','-','-','-','-','-','-']
    marks = ['','','','','','','']
    for f in files:
        tf, rmsf, divuf = extractData(f)
        time.append(tf)
        rms.append(rmsf)
        divu.append(divuf)

    plotnow('rms','$t$','$\\|V_{rms}\\|_{L2}$',time,rms,labels,linestyles=linestyles,markers=marks,ptype='semilogy')
    plotnow('div','$t$','$\\|div V\\|_{L2}$',time,divu,labels,linestyles=linestyles,markers=marks,ptype='semilogy')

    time = []
    rms = []
    divu = []
    files = ['baseline_rho1.dat','baseline_rho10.dat','baseline_rho100.dat','baseline_rho1000.dat','baseline_rho10000.dat']
    labels = ['$\\rho_r=10^0$','$\\rho_r=10^1$','$\\rho_r=10^2$','$\\rho_r=10^3$','$\\rho_r=10^4$']
    linestyles = ['--','-','-','-','-','-','-']
    marks = ['','','','','','','']
    for f in files:
        tf, rmsf, divuf = extractData(f)
        time.append(tf)
        rms.append(rmsf)
        divu.append(divuf)

    plotnow('baseline_rms','$t$','$\\|V_{rms}\\|_{L2}$',time,rms,labels,linestyles=linestyles,markers=marks,ptype='semilogy')
    plotnow('baseline_div','$t$','$\\|div V\\|_{L2}$',time,divu,labels,linestyles=linestyles,markers=marks,ptype='semilogy')

    time = []
    rms = []
    divu = []

    files = ['rho10000mu100_filtered.dat','rho10000_vsvv.dat']
    labels = ['$\\rho_r=10^4,N_{svv}=N/6$','$\\rho_r=10^4,N_{svv}=N/6 - N/2$']
    linestyles = ['-','-','-','-','-','-']
    marks = ['','','','','','','']
    for f in files:
        tf, rmsf, divuf = extractData(f)
        time.append(tf)
        rms.append(rmsf)
        divu.append(divuf)

    plotnow('rms_vsvv','$t$','$\\|V_{rms}\\|_{L2}$',time,rms,labels,linestyles=linestyles,markers=marks,ptype='semilogy')
    plotnow('div_vsvv','$t$','$\\|div V\\|_{L2}$',time,divu,labels,linestyles=linestyles,markers=marks,ptype='semilogy')
    return

if __name__=="__main__":
    starttime = time.time()
    main()
    print('--- Code ran in %s seconds ---'%(time.time()-starttime))
