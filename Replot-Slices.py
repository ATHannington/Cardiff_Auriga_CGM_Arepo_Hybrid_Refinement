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

plt.rcParams.update(matplotlib.rcParamsDefault)


# We want to utilise inplace operations to keep memory within RAM limits...
inplace = False

stack = True
DEBUG = False

singleValueKeys = ["Redshift", "Lookback", "Snap", "Rvir", "Rdisc"]
HYPARAMSPATH = "HYParams.json"
HYPARAMS = json.load(open(HYPARAMSPATH, "r"))

medianString = "50.00%"

#if "mass" not in HYPARAMS["colParams"]:
#    HYPARAMS["colParams"]+=["mass"]

if HYPARAMS["ageWindow"] is not None:
    HYPARAMS["SFRBins"] = int(math.floor(HYPARAMS["ageWindow"]/HYPARAMS["windowBins"]))
else:
    HYPARAMS["SFRBins"]  = HYPARAMS["Nbins"] 

loadPathBase = "/home/universe/c1838736/Hybrid_Refinement/"
loadDirectories = [

    # "/level4/level4_cgm/apt-figures/V2-0/spxfv/surge/level4_cgm/h5_500pc/",
    # "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_1kpc/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc/",
    # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc/",
    # # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition/",
    # # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition/",
    # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level3_cgm_almost/h5_standard/",
    # "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_standard/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_standard/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc-hy-1kpc/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc-hy-500pc/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_hy-v1/",
    # "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_hy-v2/",
    ]

styleDictGroupingKeys = {
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_standard/" : ("std","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc/" : ("surge","2kpc","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc/" : ("surge","1kpc","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc-hy-1kpc/" : ("hy","2kpc","final","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc-hy-500pc/" : ("hy","1kpc","final","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_hy-v1/" : ("hy","2kpc","V1","L5"),
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_hy-v2/" : ("hy","2kpc","V2","L5"),
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_standard/" : ("std","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level3_cgm_almost/h5_standard/" : ("std","L3"),
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/surge/level4_cgm/h5_500pc/" : ("surge","500pc","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_1kpc/" : ("surge","1kpc","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc/" : ("hy","1kpc","final","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition/" : ("hy","1kpc","hard","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition/" : ("hy","1kpc","l3-mass","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc/" : ("hy","500pc","final","L4")
}

ordering = [
    ("L5","standard"), ("L5","2kpc"), ("L5","1kpc"), ("L5","hy_v1"), ("L5","hy_v2"), ("L5","2kpc_hy_1kpc"),  ("L5","1kpc_hy_500pc"), ("L4","standard"), ("L4","1kpc"), ("L4","500pc"), ("L4","1kpc_hy_500pc"),  ("L4","1kpc_hy_500pc_l3_mass"),  ("L4","1kpc_hy_500pc_hard"), ("L4","500pc_hy_250pc"), ("L3","standard")
]

customLegendLabels = {
    ("L5","standard"):"Std. L5",
    ("L5","2kpc"):"Surge L5 +2kpc",
    ("L5","1kpc"):"Surge L5 +1kpc",
    ("L5","2kpc_hy_1kpc"):"Hy(Final) L5 +2kpc +1kpc",
    ("L5","1kpc_hy_500pc"):"Hy(Final) L5 +1kpc +500pc",
    ("L5","hy_v1"):"Hy(V1) L5 +2kpc +1kpc",
    ("L5","hy_v2"):"Hy(V2) L5 +2kpc +1kpc",
    ("L4","standard"):"Std. L4",
    ("L3","standard"):"Std. L3",
    ("L4","500pc"):"Surge L4 +500pc",
    ("L4","1kpc"):"Surge L4 +1kpc",
    ("L4","1kpc_hy_500pc"):"Hy(Final) L4 +1kpc +500pc",
    ("L4","1kpc_hy_500pc_hard"):"Hy(No Res. Transition)"+"\n"+"L4 +1kpc +500pc",
    ("L4","1kpc_hy_500pc_l3_mass"):"Hy(L3 Transition)"+"\n"+"L4 +1kpc +500pc",
    ("L4","500pc_hy_250pc"):"Hy(Final) L4 +500pc +250pc",
 }

keepPercentiles = []

simulations = []
savePaths = []
savePathsData = []

for dir in loadDirectories:
    loadpath = loadPathBase+dir+"/output/"
    simulations.append(loadpath)
    savepath = HYPARAMS["savepathfigures"] + dir + "/"
    savepathdata = loadPathBase + dir + "/"
    savePaths.append(savepath)
    savePathsData.append(savepathdata)



snapRange = [
        xx
        for xx in range(
            int(HYPARAMS["snapMin"]),
            int(HYPARAMS["snapMax"]) + 1,
            1,
        )
    ]

ylabel = {
    "T": r"T (K)",
    "R": r"R/R$_{\mathrm{200c}}$",
    "n_H": r"n$_{\mathrm{H}}$ (cm$^{-3}$)",
    "n_H_col": r"N$_{\mathrm{H}}$ (cm$^{-2}$)",
    "n_HI": r"n$_{\mathrm{HI}}$ (cm$^{-3}$)",
    "n_HI_col": r"N$_{\mathrm{HI}}$ (cm$^{-2}$)",
    "nh": r"Neutral Hydrogen Fraction",
    "B": r"|B| ($ \mu $G)",
    "vrad": r"$v_{\mathrm{r}}$ (km s$^{-1}$)",
    "vrad_in": r"$v_{\mathrm{r}}$ (km s$^{-1}$)",
    "vrad_out": r"$v_{\mathrm{r}}$ (km s$^{-1}$)",
    "gz": r"Z/Z$_{\odot}$",
    "L": r"L (kpc km s$^{-1}$)",
    "Pressure": r"P (erg cm$^{-3}$)",
    "P_thermal": r"P$_{\mathrm{Th}}$ (erg cm$^{-3}$)",
    "P_magnetic": r"P$_{\mathrm{B}}$ (erg cm$^{-3}$)",
    "P_kinetic": r"P$_{\mathrm{Kin}}$(erg cm$^{-3}$)",
    "P_tot": r"P$_{\mathrm{Tot}}$ (erg cm$^{-3}$)",
    "P_tot+k": r"P$_{\mathrm{Tot}}$ (erg cm$^{-3}$)",
    "Pthermal_Pmagnetic": r"P$_{\mathrm{Th}}$/P$_{\mathrm{B}}$",
    "P_CR": r"P$_{\mathrm{CR}}$ (erg cm$^{-3}$)",
    "PCR_Pmagnetic" : r"P$_{\mathrm{CR}}$/P$_{\mathrm{B}}$",
    "PCR_Pthermal": r"P$_{\mathrm{CR}}$/P$_{\mathrm{Th}}$",
    "gah": r"Alfvén Gas Heating (erg s$^{-1}$)",
    "bfld": r"$\mathbf{B}$ ($ \mu $G)",
    "Grad_T": r"||$\nabla$ T|| (K kpc$^{-1}$)",
    "Grad_n_H": r"||$\nabla$ n$_{\mathrm{H}}$|| (cm$^{-3}$ kpc$^{-1}$)",
    "Grad_bfld": r"||$\nabla$ $\mathrm{B}$|| ($ \mu $G kpc$^{-1}$)",
    "Grad_P_CR": r"||P$_{\mathrm{CR}}$|| (erg kpc$^{-4}$)",
    "gima" : r"SFR (M$_{\odot}$ yr$^{-1}$)",
    # "crac" : r"Alfvén CR Cooling (erg s$^{-1}$)",
    "tcool": r"t$_{\mathrm{Cool}}$ (Gyr)",
    "theat": r"t$_{\mathrm{Heat}}$ (Gyr)",
    "tcross": r"t$_{\mathrm{Sound}}$ (Gyr)",
    "tff": r"t$_{\mathrm{FF}}$ (Gyr)",
    "tcool_tff": r"t$_{\mathrm{Cool}}$/t$_{\mathrm{FF}}$",
    "csound": r"c$_{\mathrm{s}}$ (km s$^{-1}$)",
    "rho_rhomean": r"$\rho / \langle \rho \rangle$",
    "rho": r"$\rho$ (M$_{\odot}$ kpc$^{-3}$)",
    "dens": r"$\rho$  (g cm$^{-3}$)",
    "ndens": r"n (cm$^{-3}$)",
    "mass": r"Mass (M$_{\odot}$)",
    "vol": r"Volume (kpc$^{3}$)",
    "age": "Lookback Time (Gyr)",
    "cool_length" : "Cooling Length (kpc)",
    "halo" : "FoF Halo",
    "subhalo" : "SubFind Halo",
    "x": r"x (kpc)",
    "y": r"y (kpc)",
    "z": r"z (kpc)",
    "count": r"Count per pixel",
    "e_CR": r"$\epsilon_{\mathrm{CR}}$ (eV cm$^{-3}$)",
}

colImagexlimDict ={
    "n_H_col": {"xmin": 19.0, "xmax": 21.5},
    "n_HI_col" : {"xmin": 14.0, "xmax": 21.5},
    "n_H": {"xmin": -5.5, "xmax": -2.5},
    }

imageCmapDict = {
    "Pressure": "tab10",
    "vrad": "seismic",
    "vrad_out": "Reds",
    "vrad_in": "Blues",
    "n_H": (HYPARAMS["colourmapMain"].split("_"))[0],
    "n_HI": (HYPARAMS["colourmapMain"].split("_"))[0],
    "n_H_col": (HYPARAMS["colourmapMain"].split("_"))[0],
    "n_HI_col": (HYPARAMS["colourmapMain"].split("_"))[0],
}

yaxisZeroLineDict = {
    "gz": True,
    "vrad": True,
    "Pthermal_Pmagnetic": True,
    "PCR_Pthermal": True,
    "PCR_Pmagnetic": True,
}

xlimDict = {
    "R": {"xmin": 0.0, "xmax": HYPARAMS["Router"]},
    "mass": {"xmin": 4.0, "xmax": 9.0},
    "L": {"xmin": 1.5, "xmax": 4.5},
    "T": {"xmin": 4.0, "xmax": 6.5},
    "n_H": {"xmin": -6.0, "xmax": -2.5},
    "n_HI" : {"xmin": -13.0, "xmax": 0.0},
    "n_H_col": {"xmin": 19.0, "xmax": 21.5},
    "n_HI_col" : {"xmin": 12.0, "xmax": 21.5},
    "B": {"xmin": -2.5, "xmax": 1.0},
    "vrad": {"xmin": -200.0, "xmax": 200.0},
    "vrad_in": {"xmin": -200.0, "xmax": 200.0},
    "vrad_out": {"xmin": -200.0, "xmax": 200.0},
    "gz": {"xmin": -1.0, "xmax": 2.0},
    "P_thermal": {"xmin": -17.0, "xmax": -10.0},
    "P_CR": {"xmin": -17.0, "xmax": -10.0},
    "PCR_Pthermal": {"xmin": -4.0, "xmax": 1.0},
    "PCR_Pmagnetic": {"xmin": -3.0, "xmax": 3.0},
    "Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 4.0},
    "P_magnetic": {"xmin": -17.0, "xmax": -10.0},
    "P_kinetic": {"xmin": -17.0, "xmax": -10.0},
    "P_tot": {"xmin": -17.0, "xmax": -10.0},
    "P_tot+k": {"xmin": -17.0, "xmax": -10.0},
    "tcool": {"xmin": -4.0, "xmax": 4.0},
    "theat": {"xmin": -4.0, "xmax": 4.0},
    "tff": {"xmin": -1.5, "xmax": 0.75},
    "tcool_tff": {"xmin": -2.5, "xmax": 2.0},
    "rho_rhomean": {"xmin": 1.5, "xmax": 6.0},
    "dens": {"xmin": -30.0, "xmax": -22.0},
    "ndens": {"xmin": -6.0, "xmax": 0.0},
    "rho_rhomean": {"xmin": 0.25, "xmax": 6.5},
    "rho" : {"xmin": 2.0, "xmax": 7.0},
    "vol": {"xmin": -2.0, "xmax" : 3.0},
    "cool_length" : {"xmin": -1.0, "xmax": 2.0},
    "csound" : {},
    "nh" : {"xmin": -7.0, "xmax": 1.0},
    "e_CR": {"xmin": -8.0, "xmax": 0.0},
}



# ==============================================================================#
#
#          Main
#
# ==============================================================================#



for entry in HYPARAMS["logParameters"]:
    ylabel[entry] = r"$\mathrm{Log_{10}}$ " + ylabel[entry]
    ylabel[entry] = ylabel[entry].replace("(","[")
    ylabel[entry] = ylabel[entry].replace(")","]")

#   Perform forbidden log of Grad check
deleteParams = []
for entry in HYPARAMS["logParameters"]:
    entrySplit = entry.split("_")
    if (
        ("Grad" in entrySplit) &
        (np.any(np.isin(np.array(HYPARAMS["logParameters"]), np.array(
            "_".join(entrySplit[1:])))))
    ):
        deleteParams.append(entry)

for entry in deleteParams:
    HYPARAMS["logParameters"].remove(entry)


if __name__ == "__main__":
    for (loadpath,savePathBase,savePathBaseFigureData) in zip(simulations,savePaths,savePathsData):
        print(loadpath)
        # we need to nest the
        # statistics dictionaries in an outer dicitionary with some simulation descriptors, such as resolution and
        # Auriga halo number.
        splitList = loadpath.split("/")
        baseResLevel, haloLabel = splitList[-4:-2]
        tmp = haloLabel.split("_")
        haloSplitList = []
        for xx in tmp:
            splitxx = xx.split("-")
            haloSplitList += splitxx
        haloLabelKeySaveable = "_".join(haloSplitList)
        auHalo, resLabel = haloSplitList[0], "_".join(haloSplitList[1:])

        tmp = ""
        for savePathChunk in savePathBaseFigureData.split("/")[:-1]:
            tmp += savePathChunk + "/"
            try:
                os.mkdir(tmp)
            except:
                pass
            else:
                pass


        tmp = ""
        for savePathChunk in savePathBase.split("/")[:-1]:
            tmp += savePathChunk + "/"
            try:
                os.mkdir(tmp)
            except:
                pass
            else:
                pass

        for snapNumber in snapRange:
            print("\n"+
                  "\n"+f"[@{int(snapNumber)}]"
            )
            savePathFigureData = savePathBaseFigureData + "Plots/Slices/"

            if snapNumber is not None:
                if type(snapNumber) == int:
                    SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                else:
                    SaveSnapNumber = "_" + str(snapNumber)
            else:
                SaveSnapNumber = ""
    
            # Create variant of xlimDict specifically for images of col params
            tmpxlimDict = copy.deepcopy(xlimDict)

            # Add the col param specific limits to the xlimDict variant
            for key, value in colImagexlimDict.items():
                tmpxlimDict[key] = value

            #---------------#
            # Check for any none-position-based parameters we need to track for col params:
            #       Start with mass (always needed!) and xParam:
            additionalColParams = ["mass"]
            if np.any(np.isin(np.asarray([HYPARAMS["xParam"]]),np.array(["R","x","y","z","pos"]))) == False:
                additionalColParams.append(HYPARAMS["xParam"])

            #       Now add in anything we needed to track for weights of col params in statistics
            cols = HYPARAMS["colParams"]
            for param in cols:
                additionalParam = HYPARAMS["nonMassWeightDict"][param]
                if (np.any(np.isin(np.asarray([additionalParam]),np.asarray(additionalColParams))) == False) \
                & (additionalParam is not None) & (additionalParam != "count"):
                    additionalColParams.append(additionalParam)
            #---------------#
            savePathFigureData = savePathBaseFigureData + "Plots/Slices/"

            if snapNumber is not None:
                if type(snapNumber) == int:
                    SaveSnapNumber = "_" + str(snapNumber).zfill(4)
                else:
                    SaveSnapNumber = "_" + str(snapNumber)
            else:
                SaveSnapNumber = ""

            # Axes Labels to allow for adaptive axis selection
            AxesLabels = ["z","x","y"]
            colout = {}
            if len(HYPARAMS['colParams'])>0:

                # Create variant of xlimDict specifically for images of col params
                tmpxlimDict = copy.deepcopy(xlimDict)

                # Add the col param specific limits to the xlimDict variant
                for key, value in colImagexlimDict.items():
                    tmpxlimDict[key] = value

                #---------------#
                # Check for any none-position-based parameters we need to track for col params:
                #       Start with mass (always needed!) and xParam:
                additionalColParams = ["mass"]
                if np.any(np.isin(np.asarray([HYPARAMS["xParam"]]),np.array(["R","x","y","z","pos"]))) == False:
                    additionalColParams.append(HYPARAMS["xParam"])

                #       Now add in anything we needed to track for weights of col params in statistics
                cols = HYPARAMS["colParams"]
                for param in cols:
                    additionalParam = HYPARAMS["nonMassWeightDict"][param]
                    if (np.any(np.isin(np.asarray([additionalParam]),np.asarray(additionalColParams))) == False) \
                    & (additionalParam is not None) & (additionalParam != "count") & (additionalParam != "count"):
                        additionalColParams.append(additionalParam)
                #---------------#

                # If there are other params to be tracked for col params, we need to create a projection
                # of them so as to be able to map these projection values back to the col param maps.
                # A side effect of this is that we will create "images" of any of these additional params.
                # Thus, we want to provide empty limits for the colourbars of these images as they will almost
                # certainly require different limits to those provided for the PDF plots, for example. 
                # In particular, params like mass will need very different limits to those used in the
                # PDF plots. We have left this side effect in this code version as it provides a useful
                # way of testing whether making a projection of unusual params to image (e.g. mass, or volume)
                # provide sensible, physical results.
                for key in additionalColParams:
                    tmpxlimDict[key] = {}

                cols = HYPARAMS["colParams"]+additionalColParams

                for param in cols:
                    paramSplitList = param.split("_")

                    if paramSplitList[-1] == "col":
                        ## If _col variant is called we want to calculate a projection of the non-col parameter
                        ## Thus, we force projection to now be true, and incorporate a dummy variable tmpsliceParam
                        ## to force plots to generate non-col variants but save output as column density version

                        tmpsliceParam = "_".join(paramSplitList[:-1])
                        projection = True
                    else:
                        tmpsliceParam = param
                        projection = False

                        # if HYPARAMS["averageAcrossAxes"] is True:

                        #     if projection is False:
                        #         loadPathFigureData = savePathFigureData + f"Slice_Plot_AxAv_{param}{SaveSnapNumber}"
                        #     else:
                        #         loadPathFigureData = savePathFigureData + f"Projection_Plot_AxAv_{param}{SaveSnapNumber}" 

                        # else:
                        #     if projection is False:
                        #         loadPathFigureData = savePathFigureData + f"Slice_Plot_{AxesLabels[HYPARAMS['Axes'][0]]}-{AxesLabels[HYPARAMS['Axes'][1]]}_{param}{SaveSnapNumber}"
                        #     else:
                        #         loadPathFigureData = savePathFigureData + f"Projection_Plot_{AxesLabels[HYPARAMS['Axes'][0]]}-{AxesLabels[HYPARAMS['Axes'][1]]}_{param}{SaveSnapNumber}" 

                        # print("\n"+f"[@{int(snapNumber)}]: Loading {loadPathFigureData}")

                    tmpdict = apt.hy_load_individual_slice_plot_data(
                        HYPARAMS,
                        snapNumber,
                        sliceParam = param,
                        Axes = HYPARAMS["Axes"],
                        averageAcrossAxes = HYPARAMS["averageAcrossAxes"],
                        projection=[projection],
                        loadPathBase = savePathBaseFigureData,
                        loadPathSuffix = "",
                        selectKeyLen=1,
                        delimiter="-",
                        stack = None,
                        allowFindOtherAxesData = False,
                        verbose = DEBUG,
                        hush = not DEBUG
                    )
                    colout.update(tmpdict)

                    _ = apt.plot_slices(colout,
                        ylabel=ylabel,
                        xlimDict=tmpxlimDict,
                        logParameters = HYPARAMS["logParameters"],
                        snapNumber=snapNumber,
                        sliceParam = param,
                        Axes=HYPARAMS["Axes"],
                        averageAcrossAxes = HYPARAMS["averageAcrossAxes"],
                        saveAllAxesImages = HYPARAMS["saveAllAxesImages"],
                        xsize = HYPARAMS["xsizeImages"],
                        ysize = HYPARAMS["ysizeImages"],
                        colourmapMain=HYPARAMS["colourmapMain"],
                        colourmapsUnique = imageCmapDict,
                        boxsize=HYPARAMS["boxsize"],
                        boxlos=HYPARAMS["coldenslos"],
                        pixreslos=HYPARAMS["pixreslos"],
                        pixres=HYPARAMS["pixres"],
                        projection = projection,
                        DPI = HYPARAMS["DPIimages"],
                        numthreads=HYPARAMS["numthreads"],
                        savePathBase = HYPARAMS["savepathfigures"],
                        savePathBaseFigureData = HYPARAMS["savepathdata"],
                        saveFigureData = True,
                        saveFigure = True,
                        inplace = inplace,
                        replotFromData = True,
                    )

            for param in HYPARAMS["imageParams"]:
                for projection in [True,False]:        
                    tmpdict = apt.hy_load_individual_slice_plot_data(
                        HYPARAMS,
                        snapNumber,
                        sliceParam = param,
                        Axes = HYPARAMS["Axes"],
                        averageAcrossAxes = False,
                        projection=[projection],
                        loadPathBase = savePathBaseFigureData,
                        loadPathSuffix = "",
                        selectKeyLen=1,
                        delimiter="-",
                        stack = None,
                        allowFindOtherAxesData = True,
                        verbose = DEBUG,
                        hush = not DEBUG
                        )      
                    
                    _ = apt.plot_slices(
                        tmpdict,
                        ylabel=ylabel,
                        xlimDict=xlimDict,
                        logParameters = HYPARAMS["logParameters"],
                        snapNumber=snapNumber,
                        sliceParam = param,
                        Axes=HYPARAMS["Axes"],
                        xsize = HYPARAMS["xsizeImages"],
                        ysize = HYPARAMS["ysizeImages"],
                        colourmapMain = HYPARAMS["colourmapMain"],
                        colourmapsUnique = imageCmapDict,
                        boxsize=HYPARAMS["boxsize"],
                        boxlos=HYPARAMS["boxlos"],
                        pixreslos=HYPARAMS["pixreslos"],
                        pixres=HYPARAMS["pixres"],
                        projection = projection,
                        DPI = HYPARAMS["DPIimages"],
                        numthreads=HYPARAMS["numthreads"],
                        savePathBase = HYPARAMS["savepathfigures"] ,
                        savePathBaseFigureData = HYPARAMS["savepathdata"],
                        saveFigureData = True,
                        saveFigure = True,
                        inplace = inplace,
                    )

        print("finished sim:", loadpath)
    print("Finished fully! :)")
