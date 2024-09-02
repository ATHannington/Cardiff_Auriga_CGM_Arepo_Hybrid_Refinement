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

    "/level4/level4_cgm/apt-figures/V2-0/spxfv/surge/level4_cgm/h5_500pc/",
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_1kpc/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition/",
    "/level4/level4_cgm/apt-figures/V2-0/c1838736/Auriga/level3_cgm_almost/h5_standard/",
    "/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_standard/",
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_standard/",
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc/",
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc/",
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc-hy-1kpc/",
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_1kpc-hy-500pc/",
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_hy-v1/",
    "/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_hy-v2/",
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
    ("std","L5"), ("surge","2kpc","L5"), ("surge","1kpc","L5"), ("hy","2kpc","final","L5"), ("hy","1kpc","final","L5"), ("hy","2kpc","V1","L5"), ("hy","2kpc","V2","L5"), ("std","L4"), ("surge","500pc","L4"), ("surge","1kpc","L4"), ("hy","1kpc","final","L4"), ("hy","1kpc","hard","L4"), ("hy","1kpc","l3-mass","L4"), ("hy","500pc","final","L4"), ("std","L3")
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
    "T": {"xmin": 3.5, "xmax": 7.0},
    "n_H": {"xmin": -6.0, "xmax": 1.0},
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
    "ndens": {"xmin": -6.0, "xmax": 2.0},
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



    # sliceAndProjComparisonKey = [("surge","1kpc","L4")]
    # sliceAndProjComparisonPath = ["/level4/level4_cgm/apt-figures/V2-0/spxfv/Auriga/level4_cgm/h5_1kpc/" ]
    # # loadpath = HYPARAMS["savepathdata"]
    
    # tmp = apt.hy_load_slice_plot_data(
    #     sliceAndProjComparisonKey,
    #     sliceAndProjComparisonPath,
    #     PARAMS = HYPARAMSHALO,
    #     snapNumber = snapNumber,
    #     sliceParam = ["n_H","T","gz","vrad"],
    #     Axes = HYPARAMS["Axes"],
    #     projection=[False,False,False,False],
    #     loadPathBase = loadPathBase,
    #     loadPathSuffix = "",
    #     selectKeyLen=1,
    #     delimiter="-",
    #     stack = None,
    #     allowFindOtherAxesData = True,
    #     verbose = DEBUG,
    #     hush = ~ DEBUG,
    #     )

    # # orderedData = {}
    # # for key in ordering:
    # #     if key in list(tmp.keys()):
    # #         orderedData.update({key : copy.deepcopy(tmp[key])})


    # sliceAndProjComparisonKey = [("surge","2kpc","L5")]
    # sliceAndProjComparisonPath = ["/level5/level5_cgm/apt-figures/V2-0/c1838736/Auriga/level5_cgm/h5_2kpc/"]
    # # loadpath = HYPARAMS["savepathdata"]
    
    # tmp2 = apt.hy_load_slice_plot_data(
    #     sliceAndProjComparisonKey,
    #     sliceAndProjComparisonPath,
    #     PARAMS = HYPARAMSHALO,
    #     snapNumber = snapNumber,
    #     sliceParam = ["n_H","T","gz","vrad"],
    #     Axes = HYPARAMS["Axes"],
    #     projection=[False,False,False,False],
    #     loadPathBase = loadPathBase,
    #     loadPathSuffix = "",
    #     selectKeyLen=1,
    #     delimiter="-",
    #     stack = None,
    #     allowFindOtherAxesData = True,
    #     verbose = DEBUG,
    #     hush = ~ DEBUG,
    #     )
    
    # # # orderedData2 = {}
    # # for key in ordering:
    # #     if key in list(tmp2.keys()):
    # #         orderedData.update({key : copy.deepcopy(tmp2[key])})


    # orderedData = {}
    # for key in ordering:
    #     if key in list(tmp.keys()):
    #         orderedData.update({key : copy.deepcopy(tmp[key])})
    #     elif key in list(tmp2.keys()):
    #         orderedData.update({key : copy.deepcopy(tmp2[key])})

    # # variableAdjust = "2"

    # # for key,val in orderedData2.items():
    # #     tmpKey = list(key)
    # #     tmpKey2 = [variableAdjust] + tmpKey
    # #     newKey = tuple(tmpKey2)
    # #     orderedData.update({newKey: copy.deepcopy(val)})

    # tmpdict = apt.plot_slices(orderedData,
    #     ylabel=ylabel,
    #     xlimDict=xlimDict,
    #     logParameters = HYPARAMS["logParameters"],
    #     snapNumber=snapNumber,
    #     sliceParam = [["n_H","T","gz","vrad"]],
    #     Axes=HYPARAMS["Axes"],
    #     averageAcrossAxes = False,
    #     saveAllAxesImages = HYPARAMS["saveAllAxesImages"],
    #     xsize = HYPARAMS["xsizeImages"]*1.8,#*2.5,
    #     ysize = HYPARAMS["ysizeImages"]*1.8,#*1.8,*2.5,
    #     fontsize = HYPARAMS["fontsize"],
    #     colourmapMain = HYPARAMS["colourmapMain"],
    #     colourmapsUnique = imageCmapDict,
    #     boxsize = HYPARAMS["boxsize"],
    #     boxlos = HYPARAMS["boxlos"],
    #     pixreslos = HYPARAMS["pixreslos"],
    #     pixres = HYPARAMS["pixres"],
    #     projection = [[False,False,False,False]],
    #     DPI = HYPARAMS["DPI"],
    #     numthreads=HYPARAMS["numthreads"],
    #     savePathBase = HYPARAMS["savepathfigures"],
    #     savePathBaseFigureData = HYPARAMS["savepathdata"],
    #     saveFigureData = False,
    #     saveFigure = HYPARAMS["SaveImages"],
    #     selectKeysList = None,
    #     compareSelectKeysOn = "vertical",
    #     subfigures = True,
    #     subfigureDatasetLabelsBool = True,
    #     subfigureDatasetLabelsDict = {("surge","2kpc","L5"): "Surge L5 +2kpc Slice", ("surge","1kpc","L4"): "Surge L4 +1kpc Slice"},
    #     subfigureOffAlignmentAxisLabels = False,
    #     offAlignmentAxisLabels = None,
    #     inplace = False,
    #     replotFromData = True,
    # )
    
    # STOP406

    snapNumber="Averaged"

    # print(
    #     f"Time averaged Medians profile plots..."
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")     

    # tmp = apt.hy_load_statistics_data(
    #     selectKeysList,
    #     loadDirectories,
    #     snapRange,
    #     loadPathBase = loadPathBase,
    #     loadFile = "statsDict",
    #     fileType = ".h5",
    #     stack = True,
    #     verbose = DEBUG,
    #     )

    # statsOut = copy.deepcopy(tmp)    

    # if (len(snapRange)>1)&(stack is True):
    #     for sKey, data in statsOut.items():
    #         dataCopy = copy.deepcopy(data)
    #         for key,dd in data.items():
    #             for kk, value in dd.items():
    #                 dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
    #         statsOut[sKey].update(dataCopy)


    # apt.medians_versus_plot(
    #     statsOut,
    #     HYPARAMSHALO,
    #     ylabel=ylabel,
    #     xlimDict=xlimDict,
    #     snapNumber=snapNumber,
    #     yParam=HYPARAMS["mediansParams"],
    #     xParam=HYPARAMS["xParam"],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI = HYPARAMS["DPI"],
    #     xsize = HYPARAMS["xsize"],
    #     ysize = HYPARAMS["ysize"],
    #     fontsize = HYPARAMS["fontsize"],
    #     fontsizeTitle = HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     opacityPercentiles = HYPARAMS["opacityPercentiles"],
    #     savePathBase = HYPARAMS["savepathfigures"],
    #     savePathBaseFigureData = HYPARAMS["savepathdata"],
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     )


    # print(
    #     f"Time averaged Gas PDF plots..."
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")     

    # tmp = apt.hy_load_pdf_versus_plot_data(
    #     selectKeysList,
    #     loadDirectories,
    #     snapRange,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
    #     cumulative = False,
    #     loadPathBase = loadPathBase,
    #     loadPathSuffix = "",
    #     SFR = False,
    #     normalise = False,
    #     stack = True,
    #     verbose = DEBUG,
    #     )

    # pdfOut = copy.deepcopy(tmp)    

    # if (len(snapRange)>1)&(stack is True):
    #     for sKey, data in pdfOut.items():
    #         dataCopy = copy.deepcopy(data)
    #         for key,dd in data.items():
    #             for kk, value in dd.items():
    #                 dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
    #         pdfOut[sKey].update(dataCopy)

    # print(
    #     f"PDF of gas mass plot"
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")   
    
    # apt.pdf_versus_plot(
    #     pdfOut,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = [HYPARAMS["xParam"]],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["Nbins"],
    #     ageWindow=None,
    #     cumulative = False,
    #     savePathBase = HYPARAMS["savepathfigures"],
    #     savePathBaseFigureData = HYPARAMS["savepathdata"],
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = False,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = False,
    #     verbose = DEBUG,
    # )

    # print(
    #     f"Cumulative PDF of gas mass plot"
    # )

    # matplotlib.rc_file_defaults()
    # plt.close("all")  
    # apt.pdf_versus_plot(
    #     pdfOut,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = [HYPARAMS["xParam"]],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["Nbins"],
    #     ageWindow=None,
    #     cumulative = True,
    #     savePathBase = HYPARAMS["savepathfigures"],
    #     savePathBaseFigureData = HYPARAMS["savepathdata"],
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = False,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = False,
    #     verbose = DEBUG,
    # )

    # print(
    #     f"Normalised 'True' PDF of gas properties plot"
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")  
    # apt.pdf_versus_plot(
    #     pdfOut,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["Nbins"],
    #     ageWindow=None,
    #     cumulative = False,
    #     savePathBase = HYPARAMS["savepathfigures"],
    #     savePathBaseFigureData = HYPARAMS["savepathdata"],
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = False,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = True,
    #     verbose = DEBUG,        
    # )

    # print(
    #     f"Normalised Cumulative 'True' PDF of gas properties plot"
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")  
    # apt.pdf_versus_plot(
    #     pdfOut,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["Nbins"],
    #     ageWindow=None,
    #     cumulative = True,
    #     savePathBase = HYPARAMS["savepathfigures"],
    #     savePathBaseFigureData = HYPARAMS["savepathdata"],
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = False,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = True,
    #     verbose = DEBUG,
    # )


    # # # # # # -----------------------------------------------#
    # # # # # #           
    # # # # # #                     by Type
    # # # # # #
    # # # # # # -----------------------------------------------#



    # # # # # # By type plots do work, but this caused memory issues when trying to load in all particle types
    # # # # # # for our highest resolution simulations so we have commented these calls out for now. Useful diagnostic
    # # # # # # for contamination of the subhalo by different dark matter resolution types,
    # # # # # # and for ensuring any changes made to Arepo haven't broken the dark matter in an unintended way.
    # # # # # print(
    # # # # #    f"By Type PDF of mass vs R plot..."
    # # # # # )

    # # # # # apt.pdf_versus_plot(
    # # # # #    out,
    # # # # #    ylabel,
    # # # # #    xlimDict,
    # # # # #    HYPARAMS["logParameters"],
    # # # # #    snapNumber,
    # # # # #    weightKeys = HYPARAMS['nonMassWeightDict'],
    # # # # #    xParams = ["R"],
    # # # # #    savePathBase = HYPARAMS["savepathfigures"],
    # # # # #    savePathBaseFigureData = HYPARAMS["savepathdata"],
    # # # # #    saveFigureData = True,
    # # # # #    
    # # # # #    forceLogPDF = HYPARAMS["forceLogPDF"],
    # # # # # )

    # # # # # print(
    # # # # #    f"By Type Cumulative PDF of mass vs R plot..."
    # # # # # )

    # # # # # apt.pdf_versus_plot(
    # # # # #    out,
    # # # # #    ylabel,
    # # # # #    xlimDict,
    # # # # #    HYPARAMS["logParameters"],
    # # # # #    snapNumber,
    # # # # #    weightKeys = HYPARAMS['nonMassWeightDict'],
    # # # # #    xParams = ["R"],
    # # # # #    cumulative = True,
    # # # # #    savePathBase = HYPARAMS["savepathfigures"],
    # # # # #    savePathBaseFigureData = HYPARAMS["savepathdata"],
    # # # # #    saveFigureData = True,
    # # # # #    
    # # # # #    forceLogPDF = HYPARAMS["forceLogPDF"],
    # # # # # )

    # # # # # print(
    # # # # #    f"By Type Normalised Cumulative PDF of mass vs R plot..."
    # # # # # )

    # # # # # apt.pdf_versus_plot(
    # # # # #    out,
    # # # # #    ylabel,
    # # # # #    xlimDict,
    # # # # #    HYPARAMS["logParameters"],
    # # # # #    snapNumber,
    # # # # #    weightKeys = HYPARAMS['nonMassWeightDict'],
    # # # # #    xParams = ["R"],
    # # # # #    cumulative = True,
    # # # # #    normalise = True,
    # # # # #    savePathBase = HYPARAMS["savepathfigures"],
    # # # # #    savePathBaseFigureData = HYPARAMS["savepathdata"],
    # # # # #    saveFigureData = True,
    # # # # #    
    # # # # #    forceLogPDF = HYPARAMS["forceLogPDF"],
    # # # # # )







    # #-----------------------------------------------#
    # #           
    # #                     SFR
    # #
    # #-----------------------------------------------#

    # print(
    #     f"Time averaged Star properties PDF plots..."
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")     

    # tmp = apt.hy_load_pdf_versus_plot_data(
    #     selectKeysList,
    #     loadDirectories,
    #     [snapRange[-1]],
    #     weightKeys = ['gima'],
    #     xParams = ["age"],
    #     cumulative = False,
    #     loadPathBase = loadPathBase,
    #     loadPathSuffix = "",
    #     SFR = True,
    #     normalise = False,
    #     stack = True,
    #     verbose = DEBUG,
    #     )

    # pdfOutStars = copy.deepcopy(tmp)    

    # if (len(snapRange)>1)&(stack is True):
    #     for sKey, data in pdfOutStars.items():
    #         dataCopy = copy.deepcopy(data)
    #         for key,dd in data.items():
    #             for kk, value in dd.items():
    #                 dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
    #         pdfOutStars[sKey].update(dataCopy)


    # print(
    #     f"SFR plot..."
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")  
    # apt.pdf_versus_plot(
    #     pdfOutStars,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = ['gima'],
    #     xParams = ["age"],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["SFRBins"],
    #     ageWindow=HYPARAMS["ageWindow"],
    #     cumulative = False,
    #     savePathBase = HYPARAMS["savepathfigures"],
    #     savePathBaseFigureData = HYPARAMS["savepathdata"],
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = True,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = False,
    #     verbose = DEBUG,     
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")  

    # print(
    #     f"Cumulative SFR plot..."
    # )

    # apt.pdf_versus_plot(
    #     pdfOutStars,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = ['gima'],
    #     xParams = ["age"],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["SFRBins"],
    #     ageWindow=HYPARAMS["ageWindow"],
    #     cumulative = True,
    #     savePathBase = HYPARAMS["savepathfigures"],
    #     savePathBaseFigureData = HYPARAMS["savepathdata"],
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = True,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = False,
    #     verbose = DEBUG,
    # )

    # matplotlib.rc_file_defaults()
    # plt.close("all")  
    # print(
    #     f"Normalised Cumulative SFR plot..."
    # )

    # apt.pdf_versus_plot(
    #     pdfOutStars,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = ['gima'],
    #     xParams = ["age"],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["SFRBins"],
    #     ageWindow=HYPARAMS["ageWindow"],
    #     cumulative = True,
    #     savePathBase = HYPARAMS["savepathfigures"],
    #     savePathBaseFigureData = HYPARAMS["savepathdata"],
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = True,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = True,
    #     verbose = DEBUG,
    # )

    # #-----------------------------------------------#
    # #           
    # #                     CGM
    # #
    # #-----------------------------------------------#

    # print(
    #     f"PDF of CGM gas mass plot"
    # )   
    # matplotlib.rc_file_defaults()
    # plt.close("all")     

    # tmp = apt.hy_load_pdf_versus_plot_data(
    #     selectKeysList,
    #     loadDirectories,
    #     snapRange,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
    #     cumulative = False,
    #     loadPathBase = loadPathBase,
    #     loadPathSuffix = "CGM_only/",
    #     normalise = False,
    #     stack = True,
    #     verbose = DEBUG,
    #     )

    # pdfOutCGM = copy.deepcopy(tmp)    

    # if (len(snapRange)>1)&(stack is True):
    #     for sKey, data in pdfOutCGM.items():
    #         dataCopy = copy.deepcopy(data)
    #         for key,dd in data.items():
    #             for kk, value in dd.items():
    #                 dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
    #         pdfOutCGM[sKey].update(dataCopy)

    # matplotlib.rc_file_defaults()
    # plt.close("all")  
    # apt.pdf_versus_plot(
    #     pdfOutCGM,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = [HYPARAMS["xParam"]],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["Nbins"],
    #     ageWindow=None,
    #     cumulative = False,
    #     savePathBase = HYPARAMS["savepathfigures"] + "CGM_only/",
    #     savePathBaseFigureData = HYPARAMS["savepathdata"] + "CGM_only/",
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = False,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = False,
    #     verbose = DEBUG,
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")  
    # apt.pdf_versus_plot(
    #     pdfOutCGM,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = [HYPARAMS["xParam"]],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["Nbins"],
    #     ageWindow=None,
    #     cumulative = True,
    #     savePathBase = HYPARAMS["savepathfigures"] + "CGM_only/",
    #     savePathBaseFigureData = HYPARAMS["savepathdata"] + "CGM_only/",
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = False,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = False,
    #     verbose = DEBUG,
    # )

    # print(
    #     f"Normalised PDF of CGM gas plot"
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")  
    # apt.pdf_versus_plot(
    #     pdfOutCGM,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["Nbins"],
    #     ageWindow=None,
    #     cumulative = False,
    #     savePathBase = HYPARAMS["savepathfigures"] + "CGM_only/",
    #     savePathBaseFigureData = HYPARAMS["savepathdata"] + "CGM_only/",
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = False,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = True,
    #     verbose = DEBUG,        
    # )

    # print(
    #     f"Normalised Cumulative PDF of CGM gas plot"
    # )
    # matplotlib.rc_file_defaults()
    # plt.close("all")  
    # apt.pdf_versus_plot(
    #     pdfOutCGM,
    #     ylabel,
    #     xlimDict,
    #     HYPARAMS["logParameters"],
    #     snapNumber,
    #     weightKeys = HYPARAMS['nonMassWeightDict'],
    #     xParams = HYPARAMS["pdfParams"] + [HYPARAMS["xParam"]],
    #     titleBool=HYPARAMS["titleBool"],
    #     DPI=HYPARAMS["DPI"],
    #     xsize=HYPARAMS["xsize"],
    #     ysize=HYPARAMS["ysize"],
    #     fontsize=HYPARAMS["fontsize"],
    #     fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #     linewidth=HYPARAMS["linewidth"],
    #     Nbins=HYPARAMS["Nbins"],
    #     ageWindow=None,
    #     cumulative = True,
    #     savePathBase = HYPARAMS["savepathfigures"] + "CGM_only/",
    #     savePathBaseFigureData = HYPARAMS["savepathdata"] + "CGM_only/",
    #     saveFigureData = False,
    #     replotFromData = True,
    #     combineMultipleOntoAxis = True,
    #     selectKeysList = None,
    #     styleDict = styleDict,
    #     SFR = False,
    #     forceLogPDF = HYPARAMS["forceLogPDF"],
    #     normalise = True,
    #     verbose = DEBUG,
    # )

    # if len(HYPARAMS["colParams"])>0:
    #     print(
    #     "Time averaged Column Density Medians profile plots..."
    #     )

    #     selectKeysListCol = [tuple(list(sKey)+["col"]) for sKey in selectKeysList]

    #     # # # Create variant of xlimDict specifically for images of col params
    #     # # tmpxlimDict = copy.deepcopy(xlimDict)

    #     # # # Add the col param specific limits to the xlimDict variant
    #     # # for key, value in colImagexlimDict.items():
    #     # #     tmpxlimDict[key] = value

    #     #---------------#
    #     # Check for any none-position-based parameters we need to track for col params:
    #     #       Start with mass (always needed!) and xParam:
    #     additionalColParams = ["mass"]
    #     if np.any(np.isin(np.asarray([HYPARAMS["xParam"]]),np.array(["R","x","y","z","pos"]))) == False:
    #         additionalColParams.append(HYPARAMS["xParam"])

    #     #       Now add in anything we needed to track for weights of col params in statistics
    #     cols = HYPARAMS["colParams"]
    #     for param in cols:
    #         additionalParam = HYPARAMS["nonMassWeightDict"][param]
    #         if (np.any(np.isin(np.asarray([additionalParam]),np.asarray(additionalColParams))) == False) \
    #         & (additionalParam is not None) & (additionalParam != "count"):
    #             additionalColParams.append(additionalParam)
    #     #---------------#

    #     # If there are other params to be tracked for col params, we need to create a projection
    #     # of them so as to be able to map these projection values back to the col param maps.
    #     # A side effect of this is that we will create "images" of any of these additional params.
    #     # Thus, we want to provide empty limits for the colourbars of these images as they will almost
    #     # certainly require different limits to those provided for the PDF plots, for example. 
    #     # In particular, params like mass will need very different limits to those used in the
    #     # PDF plots. We have left this side effect in this code version as it provides a useful
    #     # way of testing whether making a projection of unusual params to image (e.g. mass, or volume)
    #     # provide sensible, physical results.
    #     # # for key in additionalColParams:
    #     # #     tmpxlimDict[key] = {}

    #     cols = HYPARAMS["colParams"]+additionalColParams

    #     COLHYPARAMS= copy.deepcopy(HYPARAMS)
    #     COLHYPARAMS["saveParams"]=COLHYPARAMS["saveParams"]+cols

    #     COLHYPARAMSHALO = copy.deepcopy(HYPARAMSHALO)
    #     # # COLHYPARAMSHALO = {sKey: values for sKey,(_,values) in zip(selectKeysListCol,HYPARAMSHALO.items())}
        
    #     for kk in COLHYPARAMSHALO.keys():
    #         COLHYPARAMSHALO[kk]["saveParams"] = COLHYPARAMSHALO[kk]["saveParams"]+cols

    #     matplotlib.rc_file_defaults()
    #     plt.close("all")     

    #     selectKeysListCol = [tuple(list(sKey)+["col"]) for sKey in selectKeysList]

    #     tmp = apt.hy_load_statistics_data(
    #         selectKeysListCol,
    #         loadDirectories,
    #         snapRange,
    #         loadPathBase = loadPathBase,
    #         loadFile = "colStatsDict",
    #         fileType = ".h5",
    #         stack = True,
    #         verbose = DEBUG,
    #         )

    #     statsOutCol = copy.deepcopy(tmp)    

    #     if (len(snapRange)>1)&(stack is True):
    #         for sKey, data in statsOutCol.items():
    #             dataCopy = copy.deepcopy(data)
    #             for key,dd in data.items():
    #                 for kk, value in dd.items():
    #                     dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
    #             statsOutCol[sKey].update(dataCopy)

    #     matplotlib.rc_file_defaults()
    #     plt.close("all")  
    #     apt.medians_versus_plot(
    #         statsOutCol,
    #         COLHYPARAMSHALO,
    #         ylabel=ylabel,
    #         xlimDict=xlimDict,
    #         snapNumber=snapNumber,
    #         yParam=COLHYPARAMS["colParams"],
    #         xParam=HYPARAMS["xParam"],
    #         titleBool=HYPARAMS["titleBool"],
    #         DPI = HYPARAMS["DPI"],
    #         xsize = HYPARAMS["xsize"],
    #         ysize = HYPARAMS["ysize"],
    #         fontsize = HYPARAMS["fontsize"],
    #         fontsizeTitle = HYPARAMS["fontsizeTitle"],
    #         linewidth=HYPARAMS["linewidth"],
    #         opacityPercentiles = HYPARAMS["opacityPercentiles"],
    #         savePathBase = HYPARAMS["savepathfigures"],
    #         savePathBaseFigureData = HYPARAMS["savepathdata"],
    #         saveFigureData = False,
    #         replotFromData = True,
    #         combineMultipleOntoAxis = True,
    #         selectKeysList = None,
    #         styleDict = styleDict,
    #         )
        

    #     print(
    #         f"Time averaged column density gas PDF plots..."
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all")     

    #     tmp = apt.hy_load_pdf_versus_plot_data(
    #         selectKeysListCol,
    #         loadDirectories,
    #         snapRange,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams = HYPARAMS["colParams"] + [HYPARAMS["xParam"]],
    #         cumulative = False,
    #         loadPathBase = loadPathBase,
    #         loadPathSuffix = "",
    #         SFR = False,
    #         normalise = False,
    #         stack = True,
    #         verbose = DEBUG,
    #         )

    #     pdfOutCol = copy.deepcopy(tmp)    

    #     if (len(snapRange)>1)&(stack is True):
    #         for sKey, data in pdfOutCol.items():
    #             dataCopy = copy.deepcopy(data)
    #             for key,dd in data.items():
    #                 for kk, value in dd.items():
    #                     dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
    #             pdfOutCol[sKey].update(dataCopy)

    #     print(
    #         f"PDF of column density gas mass plot"
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all") 
    #     apt.pdf_versus_plot(
    #         pdfOutCol,
    #         ylabel,
    #         xlimDict,
    #         HYPARAMS["logParameters"],
    #         snapNumber,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams = HYPARAMS["colParams"],
    #         titleBool=HYPARAMS["titleBool"],
    #         DPI=HYPARAMS["DPI"],
    #         xsize=HYPARAMS["xsize"],
    #         ysize=HYPARAMS["ysize"],
    #         fontsize=HYPARAMS["fontsize"],
    #         fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #         linewidth=HYPARAMS["linewidth"],
    #         Nbins=HYPARAMS["Nbins"],
    #         ageWindow=None,
    #         cumulative = False,
    #         savePathBase = HYPARAMS["savepathfigures"],
    #         savePathBaseFigureData = HYPARAMS["savepathdata"],
    #         saveFigureData = False,
    #         replotFromData = True,
    #         combineMultipleOntoAxis = True,
    #         selectKeysList = None,
    #         styleDict = styleDict,
    #         SFR = False,
    #         forceLogPDF = HYPARAMS["forceLogPDF"],
    #         normalise = False,
    #         verbose = DEBUG,
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all") 
    #     apt.pdf_versus_plot(
    #         pdfOutCol,
    #         ylabel,
    #         xlimDict,
    #         HYPARAMS["logParameters"],
    #         snapNumber,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams =HYPARAMS["colParams"],
    #         titleBool=HYPARAMS["titleBool"],
    #         DPI=HYPARAMS["DPI"],
    #         xsize=HYPARAMS["xsize"],
    #         ysize=HYPARAMS["ysize"],
    #         fontsize=HYPARAMS["fontsize"],
    #         fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #         linewidth=HYPARAMS["linewidth"],
    #         Nbins=HYPARAMS["Nbins"],
    #         ageWindow=None,
    #         cumulative = True,
    #         savePathBase = HYPARAMS["savepathfigures"],
    #         savePathBaseFigureData = HYPARAMS["savepathdata"],
    #         saveFigureData = False,
    #         replotFromData = True,
    #         combineMultipleOntoAxis = True,
    #         selectKeysList = None,
    #         styleDict = styleDict,
    #         SFR = False,
    #         forceLogPDF = HYPARAMS["forceLogPDF"],
    #         normalise = False,
    #         verbose = DEBUG,
    #     )

    #     print(
    #         f"Normalised PDF of column density gas plot"
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all") 
    #     apt.pdf_versus_plot(
    #         pdfOutCol,
    #         ylabel,
    #         xlimDict,
    #         HYPARAMS["logParameters"],
    #         snapNumber,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams =HYPARAMS["colParams"],
    #         titleBool=HYPARAMS["titleBool"],
    #         DPI=HYPARAMS["DPI"],
    #         xsize=HYPARAMS["xsize"],
    #         ysize=HYPARAMS["ysize"],
    #         fontsize=HYPARAMS["fontsize"],
    #         fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #         linewidth=HYPARAMS["linewidth"],
    #         Nbins=HYPARAMS["Nbins"],
    #         ageWindow=None,
    #         cumulative = False,
    #         savePathBase = HYPARAMS["savepathfigures"],
    #         savePathBaseFigureData = HYPARAMS["savepathdata"],
    #         saveFigureData = False,
    #         replotFromData = True,
    #         combineMultipleOntoAxis = True,
    #         selectKeysList = None,
    #         styleDict = styleDict,
    #         SFR = False,
    #         forceLogPDF = HYPARAMS["forceLogPDF"],
    #         normalise = True,
    #         verbose = DEBUG,        
    #     )

    #     print(
    #         f"Normalised Cumulative PDF of column density gas plot"
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all") 
    #     apt.pdf_versus_plot(
    #         pdfOutCol,
    #         ylabel,
    #         xlimDict,
    #         HYPARAMS["logParameters"],
    #         snapNumber,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams =HYPARAMS["colParams"],
    #         titleBool=HYPARAMS["titleBool"],
    #         DPI=HYPARAMS["DPI"],
    #         xsize=HYPARAMS["xsize"],
    #         ysize=HYPARAMS["ysize"],
    #         fontsize=HYPARAMS["fontsize"],
    #         fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #         linewidth=HYPARAMS["linewidth"],
    #         Nbins=HYPARAMS["Nbins"],
    #         ageWindow=None,
    #         cumulative = True,
    #         savePathBase = HYPARAMS["savepathfigures"],
    #         savePathBaseFigureData = HYPARAMS["savepathdata"],
    #         saveFigureData = False,
    #         replotFromData = True,
    #         combineMultipleOntoAxis = True,
    #         selectKeysList = None,
    #         styleDict = styleDict,
    #         SFR = False,
    #         forceLogPDF = HYPARAMS["forceLogPDF"],
    #         normalise = True,
    #         verbose = DEBUG,
    #     )

    #     print(
    #         f"Time averaged column density gas PDF plots..."
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all")     

    #     tmp = apt.hy_load_pdf_versus_plot_data(
    #         selectKeysListCol,
    #         loadDirectories,
    #         snapRange,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams = HYPARAMS["colParams"] + [HYPARAMS["xParam"]],
    #         cumulative = False,
    #         loadPathBase = loadPathBase,
    #         loadPathSuffix = "CGM_only/",
    #         SFR = False,
    #         normalise = False,
    #         stack = True,
    #         verbose = DEBUG,
    #         )

    #     pdfOutColCGM = copy.deepcopy(tmp)    

    #     if (len(snapRange)>1)&(stack is True):
    #         for sKey, data in pdfOutColCGM.items():
    #             dataCopy = copy.deepcopy(data)
    #             for key,dd in data.items():
    #                 for kk, value in dd.items():
    #                     dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
    #             pdfOutColCGM[sKey].update(dataCopy)

    #     print(
    #         f"PDF of column density gas mass plot"
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all") 
    #     apt.pdf_versus_plot(
    #         pdfOutColCGM,
    #         ylabel,
    #         xlimDict,
    #         HYPARAMS["logParameters"],
    #         snapNumber,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams = HYPARAMS["colParams"],
    #         titleBool=HYPARAMS["titleBool"],
    #         DPI=HYPARAMS["DPI"],
    #         xsize=HYPARAMS["xsize"],
    #         ysize=HYPARAMS["ysize"],
    #         fontsize=HYPARAMS["fontsize"],
    #         fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #         linewidth=HYPARAMS["linewidth"],
    #         Nbins=HYPARAMS["Nbins"],
    #         ageWindow=None,
    #         cumulative = False,
    #         savePathBase = HYPARAMS["savepathfigures"] + "CGM_only/",
    #         savePathBaseFigureData = HYPARAMS["savepathdata"] + "CGM_only/",
    #         saveFigureData = False,
    #         replotFromData = True,
    #         combineMultipleOntoAxis = True,
    #         selectKeysList = None,
    #         styleDict = styleDict,
    #         SFR = False,
    #         forceLogPDF = HYPARAMS["forceLogPDF"],
    #         normalise = False,
    #         verbose = DEBUG,
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all") 
    #     apt.pdf_versus_plot(
    #         pdfOutColCGM,
    #         ylabel,
    #         xlimDict,
    #         HYPARAMS["logParameters"],
    #         snapNumber,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams =HYPARAMS["colParams"],
    #         titleBool=HYPARAMS["titleBool"],
    #         DPI=HYPARAMS["DPI"],
    #         xsize=HYPARAMS["xsize"],
    #         ysize=HYPARAMS["ysize"],
    #         fontsize=HYPARAMS["fontsize"],
    #         fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #         linewidth=HYPARAMS["linewidth"],
    #         Nbins=HYPARAMS["Nbins"],
    #         ageWindow=None,
    #         cumulative = True,
    #         savePathBase = HYPARAMS["savepathfigures"] + "CGM_only/",
    #         savePathBaseFigureData = HYPARAMS["savepathdata"] + "CGM_only/",
    #         saveFigureData = False,
    #         replotFromData = True,
    #         combineMultipleOntoAxis = True,
    #         selectKeysList = None,
    #         styleDict = styleDict,
    #         SFR = False,
    #         forceLogPDF = HYPARAMS["forceLogPDF"],
    #         normalise = False,
    #         verbose = DEBUG,
    #     )

    #     print(
    #         f"Normalised PDF of column density gas plot"
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all") 
    #     apt.pdf_versus_plot(
    #         pdfOutColCGM,
    #         ylabel,
    #         xlimDict,
    #         HYPARAMS["logParameters"],
    #         snapNumber,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams =HYPARAMS["colParams"],
    #         titleBool=HYPARAMS["titleBool"],
    #         DPI=HYPARAMS["DPI"],
    #         xsize=HYPARAMS["xsize"],
    #         ysize=HYPARAMS["ysize"],
    #         fontsize=HYPARAMS["fontsize"],
    #         fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #         linewidth=HYPARAMS["linewidth"],
    #         Nbins=HYPARAMS["Nbins"],
    #         ageWindow=None,
    #         cumulative = False,
    #         savePathBase = HYPARAMS["savepathfigures"] + "CGM_only/",
    #         savePathBaseFigureData = HYPARAMS["savepathdata"] + "CGM_only/",
    #         saveFigureData = False,
    #         replotFromData = True,
    #         combineMultipleOntoAxis = True,
    #         selectKeysList = None,
    #         styleDict = styleDict,
    #         SFR = False,
    #         forceLogPDF = HYPARAMS["forceLogPDF"],
    #         normalise = True,
    #         verbose = DEBUG,        
    #     )

    #     print(
    #         f"Normalised Cumulative PDF of column density gas plot"
    #     )
    #     matplotlib.rc_file_defaults()
    #     plt.close("all") 
    #     apt.pdf_versus_plot(
    #         pdfOutColCGM,
    #         ylabel,
    #         xlimDict,
    #         HYPARAMS["logParameters"],
    #         snapNumber,
    #         weightKeys = HYPARAMS['nonMassWeightDict'],
    #         xParams =HYPARAMS["colParams"],
    #         titleBool=HYPARAMS["titleBool"],
    #         DPI=HYPARAMS["DPI"],
    #         xsize=HYPARAMS["xsize"],
    #         ysize=HYPARAMS["ysize"],
    #         fontsize=HYPARAMS["fontsize"],
    #         fontsizeTitle=HYPARAMS["fontsizeTitle"],
    #         linewidth=HYPARAMS["linewidth"],
    #         Nbins=HYPARAMS["Nbins"],
    #         ageWindow=None,
    #         cumulative = True,
    #         savePathBase = HYPARAMS["savepathfigures"] + "CGM_only/",
    #         savePathBaseFigureData = HYPARAMS["savepathdata"] + "CGM_only/",
    #         saveFigureData = False,
    #         replotFromData = True,
    #         combineMultipleOntoAxis = True,
    #         selectKeysList = None,
    #         styleDict = styleDict,
    #         SFR = False,
    #         forceLogPDF = HYPARAMS["forceLogPDF"],
    #         normalise = True,
    #         verbose = DEBUG,
    #     )

    print(
        "\n" + f"Time averaged gas phases plots"
    )
    matplotlib.rc_file_defaults()
    plt.close("all")     

    tmp =apt.hy_load_phase_plot_data(
        selectKeysList,
        loadDirectories,
        snapRange,
        yParams = HYPARAMS["phasesyParams"],
        xParams = HYPARAMS["phasesxParams"],
        weightKeys = HYPARAMS["phasesColourbarParams"],
        loadPathBase = loadPathBase,
        stack = True,
        verbose = DEBUG,
        )

    phaseOut = copy.deepcopy(tmp)    

    if (len(snapRange)>1)&(stack is True):
        for sKey, data in phaseOut.items():
            dataCopy = copy.deepcopy(data)
            for key,dd in data.items():
                for kk, value in dd.items():
                    dataCopy[key].update({kk: np.nanmedian(value,axis=-1)})
            phaseOut[sKey].update(dataCopy)

    # phaseOutNew = {key:val for ((_,val),key) in zip(phaseOut.items(),styleKeys)}

    for skey, dat in phaseOut.items():
        saveDir = "/".join(list(skey))
        matplotlib.rc_file_defaults()
        plt.close("all")
        apt.phase_plot(
            dat,
            ylabel,
            xlimDict,
            HYPARAMS["logParameters"],
            snapNumber = snapNumber,
            yParams = HYPARAMS["phasesyParams"],
            xParams = HYPARAMS["phasesxParams"],
            colourBarKeys = HYPARAMS["phasesColourbarParams"],
            weightKeys = HYPARAMS["nonMassWeightDict"],
            titleBool=HYPARAMS["titleBool"],
            DPI=HYPARAMS["DPI"],
            xsize=HYPARAMS["xsize"],
            ysize=HYPARAMS["ysize"],
            fontsize=HYPARAMS["fontsize"],
            fontsizeTitle=HYPARAMS["fontsizeTitle"],
            colourmapMain= HYPARAMS["colourmapMain"],
            Nbins=HYPARAMS["Nbins"],
            savePathBase = HYPARAMS["savepathfigures"]+saveDir+"/",
            savePathBaseFigureData = HYPARAMS["savepathdata"]+saveDir+"/",
            saveFigureData = False,
            verbose = DEBUG,
            inplace = inplace,
            replotFromData = True,
            allowPlotsWithoutxlimits = False,
        )

    print("Finished fully! :)")