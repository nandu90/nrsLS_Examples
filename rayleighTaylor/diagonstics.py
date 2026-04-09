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

def extract_divuerr_arrays(logfile_path):
    steps = []
    divuerrs = []

    pattern = re.compile(
              r"step\s*=\s*(\d+).*?divUErr\s+.*?([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)\s*$"
              )

    with open(logfile_path, "r", encoding="utf-8") as f:
      for line in f:
        if "divUErr" in line:
          m = pattern.search(line)
          if m:
            steps.append(int(m.group(1)))
            divuerrs.append(float(m.group(2)))  # last column

    return np.array(steps), np.array(divuerrs)

def extract_divu_nek5000(filename):
    values = []

    # regex to match floats like 1.23E-04, -2.7e+01, etc.
    float_pattern = r'[-+]?\d*\.\d+(?:[eE][-+]?\d+)?|[-+]?\d+(?:[eE][-+]?\d+)?'

    with open(filename, 'r') as f:
        for line in f:
            if "L1/L2 DIV(V)-QTL" in line:
                nums = re.findall(float_pattern, line)
                if nums:
                    # take the last number in the line
                    values.append(float(nums[-1]))

    return np.array(values)

def main():
    stepData = []
    errData = []
    labels = []
    lines = []
    marks = []

    step, diverr = extract_divuerr_arrays("logfile")
    stepData.append(step)
    errData.append(diverr)
    labels.append('nekRS')
    lines.append('--')
    marks.append('')

    diverr = extract_divu_nek5000("log5000")
    stepData.append(np.linspace(1,diverr.shape[0],diverr.shape[0]))
    errData.append(diverr)
    labels.append('nek5000')
    lines.append('--')
    marks.append('')

    plotnow('divErrs','tstep','$divUErr_{L2}$',stepData,errData,labels,linestyles=lines,markers=marks,ptype='semilogy')
    
    
    return

if __name__=="__main__":
    starttime = time.time()
    main()
    print('--- Code ran in %s seconds ---'%(time.time()-starttime))
