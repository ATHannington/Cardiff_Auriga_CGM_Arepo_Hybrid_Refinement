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

verbose = False

# We want to utilise inplace operations to keep memory within RAM limits...
inplace = True



HYPARAMSPATH = "HYParams.json"
HYPARAMS = json.load(open(HYPARAMSPATH, "r"))

if HYPARAMS["coldenspixreslos"] is None:
    HYPARAMS["coldenspixreslos"] = (HYPARAMS["coldenslos"]/HYPARAMS["boxlos"])*HYPARAMS["pixreslosproj"]

#if "mass" not in HYPARAMS["colParams"]:
#    HYPARAMS["colParams"]+=["mass"]

if HYPARAMS["ageWindow"] is not None:
    HYPARAMS["SFRBins"] = int(math.floor(HYPARAMS["ageWindow"]/HYPARAMS["windowBins"]))
else:
    HYPARAMS["SFRBins"]  = HYPARAMS["Nbins"] 

loadPathBase = "/home/cosmos/"
loadDirectories = [
    # "c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc",
    "c1838736/Auriga/spxfv/Auriga/level4_cgm/h5_500pc",
    # "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc",
    # "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition",
    # "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition",
    # "c1838736/Auriga/level3_cgm_almost/h5_standard",
    # "spxfv/Auriga/level4_cgm/h5_1kpc",
    # "spxfv/Auriga/level4_cgm/h5_standard",
    #"h5_standard",
    #"c1838736/Auriga/level5_cgm/h5_2kpc",
    #"h5_1kpc",
    #"c1838736/Auriga/level5_cgm/snapshot-restart-of-2kpc/h5_1kpc_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy-v2_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy-v3-nH_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy-v4-nH",
    # "snapshot-restart-of-2kpc/h5_hy-v5-nH",
    # "snapshot-restart-of-2kpc/h5_hy-v6-nH",
    # "snapshot-restart-of-2kpc/h5_hy-v7-ndens",
    # "snapshot-restart-of-2kpc/h5_hy-v8-ndens",
    # "snapshot-restart-of-2kpc/h5_hy-v6-ndens-ext",
    # "snapshot-restart-of-2kpc/h5_hy-v6-ndens-ext-v2",
    # "snapshot-restart-of-2kpc/h5_hy-v4-ndens-+l4",
    # "snapshot-restart-of-2kpc/h5_hy-v4-ndens-+l4-v2",
    # "snapshot-restart-of-2kpc/h5_hy-v4-ndens-+l4-v3",
    #"snapshot-restart-of-2kpc/h5_hy-v5-ndens-proper-mass-res-transition",
    #"h5_1kpc-hy-500pc",
    #"h5_2kpc-hy-1kpc",
    #"snapshot-restart-of-standard/h5_2kpc",
    #"snapshot-restart-of-standard/h5_1kpc",
    #"snapshot-restart-of-standard/h5_500pc",
    # "high-time-resolution/h5_1kpc_snapshot-restart-of-2kpc",
    # "high-time-resolution/h5_2kpc_snapshot-restart-of-2kpc",
    # "high-time-resolution/h5_hy-v2_snapshot-restart-of-2kpc",
    # "high-time-resolution/h5_1kpc_snapshot-restart-of-1kpc",
    # "high-time-resolution/h5_hy-v2_snapshot-restart-of-1kpc",
    # "h5_hy-v2",
     #"c1838736/Auriga/level5_cgm/snapshot-restart-of-2kpc/no-self-shielding/h5_1kpc_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/no-self-shielding/h5_hy-v2_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-1kpc/h5_hy-v2_snapshot-restart-of-1kpc",
    ]

simulations = []
savePaths = []
savePathsData = []

for dir in loadDirectories:
    loadpath = loadPathBase+dir+"/output/"
    simulations.append(loadpath)
    savepath = HYPARAMS["savepathfigures"] + dir + "/"
    savepathdata = HYPARAMS["savepathdata"] + dir + "/"
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
    "T": r"Temperature (K)",
    "R": r"Radius (kpc)",
    "n_H": r"n$_H$ (cm$^{-3}$)",
    "n_H_col": r"n$_H$ (cm$^{-2}$)",
    "n_HI": r"n$_{HI}$ (cm$^{-3}$)",
    "n_HI_col": r"n$_{HI}$ (cm$^{-2}$)",
    "nh": r"Neutral Hydrogen Fraction",
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
    "rho": r"Density (M$_{\odot}$ kpc$^{-3}$)",
    "dens": r"Density (g cm$^{-3}$)",
    "ndens": r"Number density (cm$^{-3}$)",
    "mass": r"Mass (M$_{\odot}$)",
    "vol": r"Volume (kpc$^{3}$)",
    "age": "Lookback Time (Gyr)",
    "cool_length" : "Cooling Length (kpc)",
}

colImagexlimDict ={
    "n_H_col": {"xmin": 19.0, "xmax": 21.5},
    "n_HI_col" : {"xmin": 14.0, "xmax": 21.5},
    "n_H": {"xmin": -5.5, "xmax": -2.5},
    }

xlimDict = {
    "R": {"xmin": 0.0, "xmax": 200.0},
    "mass": {"xmin": 4.0, "xmax": 9.0},
    "L": {"xmin": 3.0, "xmax": 4.5},
    "T": {"xmin": 3.75, "xmax": 7.0},
    "n_H": {"xmin": -6.0, "xmax": 1.0},
    "n_HI" : {"xmin": -13.0, "xmax": 0.0},
    "n_H_col": {"xmin": 19.0, "xmax": 21.5},
    "n_HI_col" : {"xmin": 12.0, "xmax": 21.5},
    "B": {"xmin": -2.5, "xmax": 1.0},
    "vrad": {"xmin": -100.0, "xmax": 100.0},
    "gz": {"xmin": -1.5, "xmax": 0.75},
    "P_thermal": {"xmin": 0.5, "xmax": 3.5},
    "P_CR": {"xmin": -1.5, "xmax": 5.5},
    "PCR_Pthermal": {"xmin": -2.0, "xmax": 2.0},
    "P_magnetic": {"xmin": -2.0, "xmax": 4.5},
    "P_kinetic": {"xmin": 0.0, "xmax": 6.0},
    "P_tot": {},#{"xmin": -1.0, "xmax": 7.0},
    "Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 10.0},
    "tcool": {},#{"xmin": -3.5, "xmax": 2.0},
    "theat": {"xmin": -4.0, "xmax": 4.0},
    "tff": {"xmin": -1.5, "xmax": 0.75},
    "tcool_tff": {"xmin": -2.5, "xmax": 2.0},
    "rho_rhomean": {"xmin": 1.5, "xmax": 6.0},
    "dens": {"xmin": -30.0, "xmax": -22.0},
    "ndens": {"xmin": -6.0, "xmax": 2.0},
    "rho_rhomean": {"xmin": 0.25, "xmax": 6.5},
    "rho" :{},
    "vol": {},
    "cool_length" : {},
    "csound" : {},
    "nh" : {"xmin": -7.0, "xmax": 1.0},
}

for entry in HYPARAMS["logParameters"]:
    ylabel[entry] = r"$Log_{10}$" + ylabel[entry]

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
                & (additionalParam is not None):
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

            colout = {}
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

                
                if projection is False:
                    loadPathFigureData = savePathFigureData + f"Slice_Plot_{param}{SaveSnapNumber}"
                else:
                    loadPathFigureData = savePathFigureData + f"Projection_Plot_{param}{SaveSnapNumber}" 

                print("\n"+f"[@{int(snapNumber)}]: Loading {loadPathFigureData}")


                out = {param: copy.deepcopy(slice)}
                colsDict = tr.hdf5_load(loadPathFigureData+"_data.h5")

                print(
                    "\n"+f"[@{int(snapNumber)}]: Re-plot {param} map..."
                )

                # By default, we set projection here to False. This ensures any weighting maps are
                # slices (projection versions were found to produce unphysical and unreliable results).
                # However, any _col parameters are forced into Projection=True inside apt.plot_slices().
                tmpdict = apt.plot_slices(colsDict,
                    ylabel=ylabel,
                    xlimDict=tmpxlimDict,
                    logParameters = HYPARAMS["logParameters"],
                    snapNumber=snapNumber,
                    sliceParam = param,
                    Axes=HYPARAMS["Axes"],
                    xsize = HYPARAMS["xsizeImages"],
                    ysize = HYPARAMS["ysizeImages"],
                    colourmapMain=HYPARAMS["colourmapMain"],
                    boxsize=HYPARAMS["boxsize"],
                    boxlos=HYPARAMS["coldenslos"],
                    pixreslos=HYPARAMS["coldenspixreslos"],
                    pixres=HYPARAMS["pixresproj"],
                    projection = projection,
                    DPI = HYPARAMS["DPIimages"],
                    numthreads=HYPARAMS["numthreads"],
                    savePathBase = savePathBase,
                    savePathBaseFigureData = savePathBaseFigureData,
                    saveFigureData = False,
                    saveFigure = True,
                    inplace = inplace,
                )
            #=============================================================#
        
            params = ["T","T","n_H","n_H","n_HI","n_HI","nh"]
            projectionBools = [True, False, True, False, True, False, False]

            for param, projection in zip(params,projectionBools):

                if projection is False:
                    loadPathFigureData = savePathFigureData + f"Slice_Plot_{param}{SaveSnapNumber}"
                    hyparamsAdjust = ""
                else:
                    loadPathFigureData = savePathFigureData + f"Projection_Plot_{param}{SaveSnapNumber}" 
                    hyparamsAdjust = "proj"

                if param == "T":
                    colourmapAdjust = "_r"
                else:
                    colourmapAdjust = ""

                print("\n"+f"[@{int(snapNumber)}]: Loading {loadPathFigureData}")


                out = {param: copy.deepcopy(slice)}
                dataDict = tr.hdf5_load(loadPathFigureData+"_data.h5")

                print(
                    "\n"+f"[@{int(snapNumber)}]: Re-plot {param} map..."
                )

                if param == "n_H":
                    limDict = tmpxlimDict
                else:
                    limDict = xlimDict

                _ = apt.plot_slices(dataDict,
                ylabel=ylabel,
                xlimDict=limDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = param,
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"]+colourmapAdjust,
                boxsize=HYPARAMS["boxsize"],
                boxlos=HYPARAMS["boxlos"],
                pixreslos=HYPARAMS["pixreslos"+hyparamsAdjust],
                pixres=HYPARAMS["pixres"+hyparamsAdjust],
                projection = projection,
                DPI = HYPARAMS["DPIimages"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
                savePathBaseFigureData = savePathBaseFigureData ,
                saveFigureData = False,
                inplace = inplace,
                )
           
        print("finished sim:", loadpath)
    print("Finished fully! :)")
