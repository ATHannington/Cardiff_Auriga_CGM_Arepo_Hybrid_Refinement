import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
import matplotlib.transforms as tx
from matplotlib.ticker import AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import cm
import const as c
import OtherConstants as oc
from gadget import *
from gadget_subfind import *
import Tracers_Subroutines as tr
import CR_Subroutines as cr
import Plotting_tools as apt
import h5py
import json
import copy
import math
import os

ageWindow = None #(Gyr) before current snapshot SFR evaluation
windowBins = 0.100 #(Gyr) size of ageWindow Bins. Ignored if ageWindow is None
Nbins = 250
snapStart = 100
snapEnd = 127 #116
DEBUG = False
forceLogMass = False
numthreads = 18

loadPathBase = "/home/cosmos/"
loadDirectories = [
    "spxfv/Auriga/level4_cgm/h5_standard",
    "spxfv/Auriga/level4_cgm/h5_1kpc",
    "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc",
    "c1838736/Auriga/spxfv/Auriga/level4_cgm/h5_500pc",
    "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition",
    "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition",
    "c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc",
    "c1838736/Auriga/level3_cgm_almost/h5_standard",
    #"h5_standard",
    # "h5_2kpc",
    # #"h5_1kpc",
    # "snapshot-restart-of-2kpc/h5_1kpc_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy-v2_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy-v3-nH_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy-v4-nH",
    # "snapshot-restart-of-2kpc/h5_hy-v5-nH",
    # "snapshot-restart-of-2kpc/h5_hy-v6-nH",
    # "snapshot-restart-of-2kpc/h5_hy-v7-ndens",
    # "snapshot-restart-of-2kpc/h5_hy-v8-ndens",
    # "snapshot-restart-of-2kpc/h5_hy-v6-ndens-ext",
    #"snapshot-restart-of-2kpc/h5_hy-v6-ndens-ext-v2",
    #"snapshot-restart-of-2kpc/h5_hy-v5-ndens-proper-mass-res-transition",
    #"snapshot-restart-of-2kpc/h5_hy-v4-ndens-+l4-v3",
    #"h5_2kpc-hy-1kpc",
    #"h5_1kpc-hy-500pc",
    # "h5_standard",
    # "h5_1kpc",
    # "snapshot-restart-of-1kpc/h5_hy-v2_snapshot-restart-of-1kpc",
    # "h5_standard",
    # "h5_2kpc",
    # "h5_1kpc",
    # "h5_hy-v2",
    # "snapshot-restart-of-2kpc/no-self-shielding/h5_1kpc_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/no-self-shielding/h5_hy-v2_snapshot-restart-of-2kpc",
]

simulations = []
savePaths = []

for dir in loadDirectories:
    loadpath = loadPathBase+dir+"/output/"
    simulations.append(loadpath)

    savepath = "./" + dir + "/"
    savePaths.append(savepath)

if ageWindow is not None:
    SFRBins = int(math.floor(ageWindow/windowBins))
else:
    SFRBins = Nbins


snapRange = [
        xx
        for xx in range(
            int(snapStart),
            int(snapEnd) + 1,
            1,
        )
    ]

ylabel = {
    "T": r"Temperature (K)",
    "R": r"Radius (kpc)",
    "n_H": r"n$_H$ (cm$^{-3}$)",
    "B": r"|B| ($ \mu $G)",
    "vrad": r"Radial Velocity (km s$^{-1}$)",
    "gz": r"Metallicity Z$_{\odot}$",
    "L": r"Specific Angular Momentum" + "\n" + r"(kpc km s$^{-1}$)",
    "P_thermal": r"P$_{Thermal}$ / k$_B$ (K cm$^{-3}$)",
    "P_magnetic": r"P$_{Magnetic}$ / k$_B$ (K cm$^{-3}$)",
    "P_kinetic": r"P$_{Kinetic}$ / k$_B$ (K cm$^{-3}$)",
    "P_tot": r"P$_{tot}$ = (P$_{thermal}$ + P$_{magnetic}$)/ k$_B$"
    + "\n"
    + r"(K cm$^{-3}$)",
    "Pthermal_Pmagnetic": r"P$_{thermal}$/P$_{magnetic}$",
    "P_CR": r"P$_{CR}$ (K cm$^{-3}$)",
    "PCR_Pthermal": r"(X$_{CR}$ = P$_{CR}$/P$_{Thermal}$)",
    "gah": r"Alfven Gas Heating (erg s$^{-1}$)",
    "bfld": r"||B-Field|| ($ \mu $G)",
    "Grad_T": r"||Temperature Gradient|| (K kpc$^{-1}$)",
    "Grad_n_H": r"||n$_H$ Gradient|| (cm$^{-3}$ kpc$^{-1}$)",
    "Grad_bfld": r"||B-Field Gradient|| ($ \mu $G kpc$^{-1}$)",
    "Grad_P_CR": r"||P$_{CR}$ Gradient|| (K kpc$^{-4}$)",
    "gima" : r"Star Formation Rate (M$_{\odot}$ yr$^{-1}$)",
    # "crac" : r"Alfven CR Cooling (erg s$^{-1}$)",
    "tcool": r"Cooling Time (Gyr)",
    "theat": r"Heating Time (Gyr)",
    "tcross": r"Sound Crossing Cell Time (Gyr)",
    "tff": r"Free Fall Time (Gyr)",
    "tcool_tff": r"t$_{Cool}$/t$_{FreeFall}$",
    "csound": r"Sound Speed (km s$^{-1}$)",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "dens": r"Density (g cm$^{-3}$)",
    "ndens": r"Number density (cm$^{-3}$)",
    "mass": r"Mass (M$_{\odot}$)",
    "vol": r"Volume (kpc$^{3}$)",
    "age": "Lookback Time (Gyr)",
    "cool_length" : "Cooling Length (kpc)",
}

xlimDict = {
    "R": {"xmin": 0.0, "xmax": 175.0},
    "mass": {"xmin": 4.0, "xmax": 9.0},
    "L": {"xmin": 3.0, "xmax": 4.5},
    "T": {"xmin": 3.75, "xmax": 7.0},
    "n_H": {"xmin": -5.5, "xmax": -0.5},
    "B": {"xmin": -2.5, "xmax": 1.0},
    "vrad": {"xmin": -100.0, "xmax": 100.0},
    "gz": {"xmin": -1.5, "xmax": 0.75},
    "P_thermal": {"xmin": 0.5, "xmax": 3.5},
    "P_CR": {"xmin": -1.5, "xmax": 5.5},
    "PCR_Pthermal": {"xmin": -2.0, "xmax": 2.0},
    "P_magnetic": {"xmin": -2.0, "xmax": 4.5},
    "P_kinetic": {"xmin": 0.0, "xmax": 6.0},
    "P_tot": {"xmin": -1.0, "xmax": 7.0},
    "Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 10.0},
    "tcool": {"xmin": -3.5, "xmax": 2.0},
    "theat": {"xmin": -4.0, "xmax": 4.0},
    "tff": {"xmin": -1.5, "xmax": 0.75},
    "tcool_tff": {"xmin": -2.5, "xmax": 2.0},
    "rho_rhomean": {"xmin": 1.5, "xmax": 6.0},
    "dens": {"xmin": -30.0, "xmax": -22.0},
    "ndens": {"xmin": -6.0, "xmax": 2.0},
    "rho_rhomean": {"xmin": 0.25, "xmax": 6.5},
    "vol": {},#{"xmin": 0.5**4, "xmax": 4.0**4}
    "mass-pdf": {},#{"xmin": 0.0, "xmax": 1e13},
    "cool_length" : {},#{"xmin": -0.5, "xmax": 1.0},
}

logParameters = ["dens","ndens","rho_rhomean","csound","T","n_H","B","gz","L","P_thermal","P_magnetic","P_kinetic","P_tot","Pthermal_Pmagnetic", "P_CR", "PCR_Pthermal","gah","Grad_T","Grad_n_H","Grad_bfld","Grad_P_CR","tcool","theat","tcross","tff","tcool_tff","mass","vol","cool_length"] #"gima"


for entry in logParameters:
    ylabel[entry] = r"$Log_{10}$" + ylabel[entry]

#   Perform forbidden log of Grad check
deleteParams = []
for entry in logParameters:
    entrySplit = entry.split("_")
    if (
        ("Grad" in entrySplit) &
        (np.any(np.isin(np.array(logParameters), np.array(
            "_".join(entrySplit[1:])))))
    ):
        deleteParams.append(entry)

for entry in deleteParams:
    logParameters.remove(entry)


if __name__ == "__main__":
    for snapNumber in snapRange:
        print(
            f"[@{int(snapNumber)}]: PDF of mass vs R plot..."
        )

        apt.combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["R"],
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: Cumulative PDF of mass vs R plot..."
        )

        apt.combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["R"],
            cumulative = True,
            forceLogMass = forceLogMass,

        )

        print(
            f"[@{int(snapNumber)}]: Normalised Cumulative PDF of mass vs R plot..."
        )

        apt.combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["R"],
            cumulative = True,
            normalise = True,
            forceLogMass = forceLogMass,

        )

        #print(
        #    f"[@{int(snapNumber)}]: By Type PDF of mass vs R plot..."
        #)

        #apt.combined_pdf_versus_plot(
        #    savePaths,
        #    ylabel,
        #    xlimDict,
        #    logParameters,
        #    snapNumber,
        #    weightKeys = ['mass'],
        #    xParams = ["R"],
        #    byType = True,
        #    forceLogMass = forceLogMass,
        #)

        #print(
        #    f"[@{int(snapNumber)}]: By Type Cumulative PDF of mass vs R plot..."
        #)

        #apt.combined_pdf_versus_plot(
        #    savePaths,
        #    ylabel,
        #    xlimDict,
        #    logParameters,
        #    snapNumber,
        #    weightKeys = ['mass'],
        #    xParams = ["R"],
        #    cumulative = True,
        #    byType = True,
        #    forceLogMass = forceLogMass,
        #)

        #print(
        #    f"[@{int(snapNumber)}]: By Type Normalised Cumulative PDF of mass vs R plot..."
        #)

        #apt.combined_pdf_versus_plot(
        #    savePaths,
        #    ylabel,
        #    xlimDict,
        #    logParameters,
        #    snapNumber,
        #    weightKeys = ['mass'],
        #    xParams = ["R"],
        #    cumulative = True,
        #    byType = True,
        #    normalise = True,
        #    forceLogMass = forceLogMass,
        #)

        print(
            f"[@{int(snapNumber)}]: SFR plot..."
        )

        apt.combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['gima'],
            xParams = ["age"],
            SFR = True,
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: Cumulative SFR plot..."
        )

        apt.combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['gima'],
            xParams = ["age"],
            cumulative = True,
            SFR = True,
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: Normalised Cumulative SFR plot..."
        )

        apt.combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['gima'],
            xParams = ["age"],
            cumulative = True,
            normalise = True,
            SFR = True,
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: PDF of gas (mass vs T or vol) plot"
        )


        apt.combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["T","vol","n_H"],
            forceLogMass = forceLogMass,
        )


        print(
            f"[@{int(snapNumber)}]: Cumulative PDF of gas (mass vs T or vol) plot"

        )
        apt.combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["T","vol","n_H"],
            cumulative = True,
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: Normalised Cumulative PDF of gas (mass vs T or vol) plot"

        )
        apt.combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["T","vol","n_H"],
            cumulative = True,
            normalise = True,
            forceLogMass = forceLogMass,
        )

    print("***")
    print("...done!")
    print("***")
