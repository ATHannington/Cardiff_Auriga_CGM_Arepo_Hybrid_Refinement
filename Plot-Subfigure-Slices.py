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
DEBUG = True

HYPARAMSPATH = "HYParams.json"
HYPARAMS = json.load(open(HYPARAMSPATH, "r"))


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
    # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc/",
    # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc/",
    # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition/",
    # "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level3_cgm_almost/h5_standard/",
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_standard/",
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_standard/",
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
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/surge/level4_cgm/h5_500pc/" : ("surge","500pc","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_1kpc/" : ("surge","1kpc","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc/" : ("hy","1kpc","final","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition/" : ("hy","1kpc","hard","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition/" : ("hy","1kpc","l3-mass","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc/" : ("hy","500pc","final","L4"),
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level3_cgm_almost/h5_standard/" : ("std","L3")
}

ordering = [
    ("std","L5"), ("surge","2kpc","L5"), ("surge","1kpc","L5"), ("hy","2kpc","final","L5"), ("hy","1kpc","final","L5"), ("hy","2kpc","V1","L5"), ("hy","2kpc","V2","L5"), ("std","L4"), ("surge","1kpc","L4"), ("hy","1kpc","final","L4"), ("hy","1kpc","hard","L4"), ("hy","1kpc","l3-mass","L4") ,("surge","500pc","L4"), ("hy","500pc","final","L4"), ("std","L3")
]
keepPercentiles = []

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
    "T": r"T (K)",
    "R": r"R (kpc)",
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
    "L": r"L" + "\n" + r"(kpc km s$^{-1}$)",
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
xlimDict = {
    "R": {"xmin": 0.0, "xmax": HYPARAMS["Router"]},
    "mass": {"xmin": 4.0, "xmax": 9.0},
    "L": {"xmin": 1.5, "xmax": 4.5},
    "T": {"xmin": 3.5, "xmax": 7.0},
    "n_H": {"xmin": -6.0, "xmax": 1.0},
    "n_HI" : {"xmin": -13.0, "xmax": 0.0},
    "n_H_col": {"xmin": 19.0, "xmax": 21.5},
    "n_HI_col" : {"xmin": 12.0, "xmax": 21.5},
    "B": {"xmin": -2.5, "xmax": 1.0},
    "vrad": {"xmin": -200.0, "xmax": 200.0},
    "vrad_in": {"xmin": -200.0, "xmax": 200.0},
    "vrad_out": {"xmin": -200.0, "xmax": 200.0},
    "gz": {"xmin": -2.0, "xmax": 1.0},
    "P_thermal": {"xmin": -19.5, "xmax": -10.0},
    "P_CR": {"xmin": -19.5, "xmax": -10.0},
    "PCR_Pthermal": {"xmin": -4.0, "xmax": 1.0},
    "PCR_Pmagnetic": {"xmin": -3.0, "xmax": 3.0},
    "Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 4.0},
    "P_magnetic": {"xmin": -19.5, "xmax": -10.0},
    "P_kinetic": {"xmin": -19.5, "xmax": -10.0},
    "P_tot": {"xmin": -19.5, "xmax": -10.0},
    "P_tot+k": {"xmin": -19.5, "xmax": -10.0},
    "tcool": {"xmin": -4.0, "xmax": 4.0},
    "theat": {"xmin": -4.0, "xmax": 4.0},
    "tff": {"xmin": -1.5, "xmax": 0.75},
    "tcool_tff": {"xmin": -2.5, "xmax": 2.0},
    "rho_rhomean": {"xmin": 1.5, "xmax": 6.0},
    "dens": {"xmin": -30.0, "xmax": -22.0},
    "ndens": {"xmin": -6.0, "xmax": 2.0},
    "rho_rhomean": {"xmin": 0.25, "xmax": 6.5},
    "rho" : {"xmin": 2.0, "xmax": 7.0},
    "vol": {"xmin": -1.0, "xmax" : 0.5},
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


    # # # loadDirectories = [
    # # # "/c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc/",
    # # # "/spxfv/surge/level4_cgm/h5_500pc/",
    # # # "/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc/",
    # # # "/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition/",
    # # # "/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition/",
    # # # "/c1838736/Auriga/level3_cgm_almost/h5_standard/",
    # # # "/spxfv/Auriga/level4_cgm/h5_standard/",
    # # # "/spxfv/Auriga/level4_cgm/h5_1kpc/",
    # # # ]

    selectKeysList = []
    HYPARAMSHALO = {}
    styleKeys = []
    for (loadpath,savePathBase,savePathBaseFigureData) in zip(loadDirectories,savePaths,savePathsData):
        print(loadpath)
        # we need to nest the
        # statistics dictionaries in an outer dicitionary with some simulation descriptors, such as resolution and
        # Auriga halo number.
        splitList = loadpath.split("/")
        baseResLevel, haloLabel = splitList[-3:-1]
        baseAdjusted = "L"+(baseResLevel.split("_"))[0][-1]
        tmp = haloLabel.split("_")
        haloSplitList = []
        for xx in tmp:
            splitxx = xx.split("-")
            haloSplitList += splitxx
        haloLabelKeySaveable = "_".join(haloSplitList)
        auHalo, resLabel = haloSplitList[0], "_".join([ll for ll in haloSplitList[1:] if (ll!="transition")&(ll!="res")])

        selectKey = (baseAdjusted, resLabel)
        selectKeysList.append(selectKey)
        HYPARAMSHALO.update({selectKey: HYPARAMS})
        styleKeys.append(styleDictGroupingKeys[loadpath])

    # ----------------------------------------------------------------------#
    #  Plots...
    # ----------------------------------------------------------------------#
    


    tmpstyleDict = apt.get_linestyles_and_colours(styleKeys,colourmapMain="tab10",colourGroupBy=[],linestyleGroupBy=["hy","surge","std"],lastColourOffset = 0.0)
    
    styleDict = {}

    for selectKey,dd in zip(selectKeysList,tmpstyleDict.values()):
        styleDict.update({selectKey : dd})


    snapNumber=snapRange[-1]

    sliceAndProjComparisonKey = [("std","L5")]
    sliceAndProjComparisonPath =  ["/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_standard/"]

    # loadpath = HYPARAMS["savepathdata"]
    
    tmp = apt.hy_load_slice_plot_data(
        sliceAndProjComparisonKey,
        sliceAndProjComparisonPath,
        PARAMS = HYPARAMSHALO,
        snapNumber = snapNumber,
        sliceParam = ["T","n_H"],
        Axes = HYPARAMS["Axes"][-1::-1],
        projection=[False,False],
        loadPathBase = loadPathBase,
        loadPathSuffix = "",
        selectKeyLen=1,
        delimiter="-",
        stack = None,
        allowFindOtherAxesData = True,
        allowNoAxes = True,
        verbose = DEBUG,
        hush = ~ DEBUG,
        )

    orderedData = {}
    for key in ordering:
        if key in list(tmp.keys()):
            orderedData.update({key : copy.deepcopy(tmp[key])})

 
    sliceAndProjComparisonKey = [("std","L4")]
    sliceAndProjComparisonPath =  ["/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_standard/"]
    # loadpath = HYPARAMS["savepathdata"]
    
    tmp2 = apt.hy_load_slice_plot_data(
        sliceAndProjComparisonKey,
        sliceAndProjComparisonPath,
        PARAMS = HYPARAMSHALO,
        snapNumber = snapNumber,
        sliceParam = ["T","n_H"],
        Axes = HYPARAMS["Axes"][-1::-1],
        projection=[False,False],
        loadPathBase = loadPathBase,
        loadPathSuffix = "",
        selectKeyLen=1,
        delimiter="-",
        stack = None,
        allowFindOtherAxesData = True,
        allowNoAxes = True,
        verbose = DEBUG,
        hush = ~ DEBUG,
        )
    
    orderedData = {}
    for key in ordering:
        if key in list(tmp2.keys()):
            orderedData.update({key : copy.deepcopy(tmp2[key])})
    
    sliceAndProjComparisonKey = [("std","L3")]
    sliceAndProjComparisonPath =  ["/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level3_cgm_almost/h5_standard/" ]

    loadpath = HYPARAMS["savepathdata"]
    
    tmp3 = apt.hy_load_slice_plot_data(
        sliceAndProjComparisonKey,
        sliceAndProjComparisonPath,
        PARAMS = HYPARAMSHALO,
        snapNumber = snapNumber,
        sliceParam = ["T","n_H"],
        Axes = HYPARAMS["Axes"][-1::-1],
        projection=[False,False],
        loadPathBase = loadPathBase,
        loadPathSuffix = "",
        selectKeyLen=1,
        delimiter="-",
        stack = None,
        allowFindOtherAxesData = True,
        allowNoAxes = True,
        verbose = DEBUG,
        hush = ~ DEBUG,
        )


    orderedData = {}
    for key in ordering:
        if key in list(tmp.keys()):
            orderedData.update({key : copy.deepcopy(tmp[key])})
        elif key in list(tmp2.keys()):
            orderedData.update({key : copy.deepcopy(tmp2[key])})
        elif key in list(tmp3.keys()):
            orderedData.update({key : copy.deepcopy(tmp3[key])})
    # variableAdjust = "2"

    # # for key,val in orderedData2.items():
    # #     tmpKey = list(key)
    # #     tmpKey2 = [variableAdjust] + tmpKey
    # #     newKey = tuple(tmpKey2)
    # #     orderedData.update({newKey: copy.deepcopy(val)})

    tmpdict = apt.plot_slices(orderedData,
        ylabel=ylabel,
        xlimDict=xlimDict,
        logParameters = HYPARAMS["logParameters"],
        snapNumber=snapNumber,
        sliceParam = [["T","n_H"]],
        Axes=HYPARAMS["Axes"],
        averageAcrossAxes = False,
        saveAllAxesImages = HYPARAMS["saveAllAxesImages"],
        xsize = HYPARAMS["xsizeImages"]*1.8,#2.8,#*1.8,#*2.5,
        ysize = HYPARAMS["ysizeImages"]*1.8,#2.8,#*1.8,#*1.8,*2.5,
        fontsize = HYPARAMS["fontsize"]*1.5,#2.5,
        colourmapMain = HYPARAMS["colourmapMain"],
        colourmapsUnique = imageCmapDict,
        boxsize = HYPARAMS["boxsize"],
        boxlos = HYPARAMS["boxlos"],
        pixreslos = HYPARAMS["pixreslos"],
        pixres = HYPARAMS["pixres"],
        projection = [[False,False]],
        DPI = HYPARAMS["DPI"],
        numthreads=HYPARAMS["numthreads"],
        savePathBase = HYPARAMS["savepathfigures"],
        savePathBaseFigureData = HYPARAMS["savepathdata"],
        saveFigureData = False,
        saveFigure = HYPARAMS["SaveImages"],
        selectKeysList = None,
        compareSelectKeysOn = HYPARAMS["compareSelectKeysOn"],
        subfigures = True,
        subfigureDatasetLabelsBool = True,
        subfigureDatasetLabelsDict = {("std","L5"):"Std. L5 Slice",("std","L4"):"Std. L4 Slice",("std","L3"):"Std. L3 Slice"},#("surge","2kpc","L5"):"Surge L5 +2kpc Slice"},#,("surge","500pc","L4"):"Surge L4 +500pc Slice"},
        subfigureOffAlignmentAxisLabels = False,
        offAlignmentAxisLabels = None,
        cbarscale = 0.3,#0.4,
        inplace = False,
        replotFromData = True,
    ) 
# ("hy","500pc","final","L4"): "Hy(final) L4 +500pc +250pc Slice"},
    print("Finished fully! :)")