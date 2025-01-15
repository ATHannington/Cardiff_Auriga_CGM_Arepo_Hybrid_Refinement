# coding=utf-8
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # For suppressing plotting on clusters
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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
import os
import math

# =============================================================================#
#
#               USER DEFINED PARAMETERS
#
# ==============================================================================#

matplotlib.use("Agg")  # For suppressing plotting on clusters

DEBUG = False
inplace = True
HYPARAMSPATHMASTER = "HYParams_Quantitative-Summary.json"

singleValueKeys = ["Redshift", "Lookback", "Snap", "Rvir", "Rdisc"]


HYPARAMS = json.load(open(HYPARAMSPATHMASTER, "r"))

if HYPARAMS["ageWindow"] is not None:
    HYPARAMS["SFRBins"] = int(math.floor(HYPARAMS["ageWindow"]/HYPARAMS["windowBins"]))
else:
    HYPARAMS["SFRBins"]  = HYPARAMS["Nbins"] 

loadPathBase = "/home/tango/" # "/home/cosmos/" # 
loadDirectories = [
    # "spxfv/Auriga/level4_cgm/h5_standard",
    # "c1838736/Auriga/level3_cgm_almost/h5_standard",
    # "spxfv/Auriga/level4_cgm/h5_1kpc",
    # "c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc",
    "spxfv/surge/level4_cgm/h5_500pc",
    # "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc",
    # "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition",
    # "c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition",
    # "c1838736/Auriga/level5_cgm/h5_standard",
    # "c1838736/Auriga/level5_cgm/h5_2kpc",
    # "c1838736/Auriga/level5_cgm/h5_1kpc",
    # "c1838736/Auriga/level5_cgm/h5_2kpc-hy-1kpc",
    # "c1838736/Auriga/level5_cgm/h5_1kpc-hy-500pc",
    # "c1838736/Auriga/level5_cgm/h5_hy-v1/",
    # "c1838736/Auriga/level5_cgm/h5_hy-v2",
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
    "T": r"T (K)",
    "R": r"R/R$_{\mathrm{200c}}}$",
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
    "R": {"xmin": 0, "xmax": HYPARAMS["Router"]},
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
    "P_thermal": {"xmin": -16.0, "xmax": -10.0},
    "P_CR": {"xmin": -19.5, "xmax": -10.0},
    "PCR_Pthermal": {"xmin": -4.5, "xmax": 2.5},
    "PCR_Pmagnetic": {"xmin": -3.5, "xmax": 2.5},
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
    "vol": {"xmin": -2.0, "xmax": 0.5},
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


# ==============================================================================#
#
#          Main
#
# ==============================================================================#


def err_catcher(arg):
    raise Exception(f"Child Process died and gave error: {arg}")
    return


snapRange = [
    xx
    for xx in range(
        int(HYPARAMS["snapMin"]),
        int(HYPARAMS["snapMax"]) + 1,
        1,
    )
]

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

        runAnalysisBool = True
        DataSavepathBase = HYPARAMS["savepathdata"]
        FigureSavepathBase = HYPARAMS["savepathfigures"]
        lastSnapStarsDict = {}
        CGMgasDict = {}
        colDict = {}
        starsDict = {}
        dataDict = {}
        fullDataDict = {}
        finalOut = {}

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

        if (HYPARAMS["loadRotationMatrix"] == True) & (HYPARAMS["constantRotationMatrix"] == True):
            rotationloadpath = savePathBaseFigureData + f"rotation_matrix_{int(snapNumber)}.h5"
            tmp = tr.hdf5_load(rotationloadpath)
            rotation_matrix = tmp[(baseResLevel, haloLabelKeySaveable)]["rotation_matrix"]
            print(
                "\n" + f"Loaded rotation_matrxix : "+
                "\n" + f"{(baseResLevel, haloLabelKeySaveable)} : 'rotation_matrix': ..."+
                "\n" + f"from {rotationloadpath}"
            )      
        else:
            rotation_matrix = None
 


        for snapNumber in snapRange:
            # rotation_matrix = None
            # snapNumber = 100
            # loadPathBase = "/home/cosmos/c1838736/Auriga/level5_cgm/"
            # simulation = "h5_2kpc"
            # loadpath = loadPathBase+simulation+"/output/"
            print(f"[@{int(snapNumber)}]: Load subfind")
            # load in the subfind group files
            snap_subfind = load_subfind(snapNumber, dir=loadpath)

            print(f"[@{int(snapNumber)}]: Load snapshot")
            snap = gadget_readsnap(
                snapNumber,
                loadpath,
                hdf5=True,
                loadonlytype=[0,1,4,5],#[0, 1, 2, 3, 4, 5],
                lazy_load=True,
                subfind=snap_subfind,
                loadonlyhalo=int(HYPARAMS["HaloID"]),
            )


            print(f"[@{int(snapNumber)}]: Rotate and centre snapshot")
            snap.calc_sf_indizes(snap_subfind)
            if rotation_matrix is None:
                print(f"[@{int(snapNumber)}]: New rotation of snapshots")
                rotation_matrix = snap.select_halo(snap_subfind, do_rotation=True)
                rotationsavepath = savePathBaseFigureData + f"rotation_matrix_{int(snapNumber)}.h5"
                tr.hdf5_save(rotationsavepath,{(baseResLevel, haloLabelKeySaveable) : {"rotation_matrix": rotation_matrix}})
                print(
                    "\n" + f"[@{int(snapNumber)}]: Saved rotation_matrxix as"+
                    "\n" + f"{(baseResLevel, haloLabelKeySaveable)} : 'rotation_matrix': ..."+
                    "\n" + f"at {rotationsavepath}"
                )
                ## If we don't want to use the same rotation matrix for all snapshots, set rotation_matrix back to None
                if (HYPARAMS["constantRotationMatrix"] == False):
                    rotation_matrix = None
            else:
                print(f"[@{int(snapNumber)}]: Existing rotation of snapshots")
                snap.select_halo(snap_subfind, do_rotation=False)
                snap.rotateto(
                    rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
                )


            print(
                f"[@{int(snapNumber)}]: SnapShot loaded at RedShift z={snap.redshift:0.05e}"
            )
            
            print(
                f"[@{int(snapNumber)}]: Clean SnapShot parameters..."
            )

            snap = cr.clean_snap_params(
                snap,
                paramsOfInterest = HYPARAMS["saveParams"] + HYPARAMS["saveEssentials"]
            )

            # --------------------------#
            ##    Units Conversion    ##
            # --------------------------#

            # Convert Units
            ## Make this a seperate function at some point??
            snap.pos *= 1e3  # [kpc]
            snap.vol *= 1e9  # [kpc^3]
            snap.mass *= 1e10  # [Msol]
            snap.hrgm *= 1e10  # [Msol]
            snap.gima *= 1e10  # [Msol]

            snap.data["R"] = np.linalg.norm(snap.data["pos"], axis=1)
            rvir = (snap_subfind.data["frc2"] * 1e3)[int(0)]
            boxmax = max([HYPARAMS['boxsize'],HYPARAMS['boxlos'],HYPARAMS['coldenslos']])
            stellarType = 4
            rdisc = (snap_subfind.data["shmt"] * 1e3)[HYPARAMS["HaloID"]][stellarType]
            

            xlimDict.update({"xmin": 0})
            xlimDict.update({"xmax": HYPARAMS["Router"]*rvir})

            print(
                f"[@{int(snapNumber)}]: Remove beyond {boxmax:2.2f} kpc..."
            )

            ## For images we want to expand boxmax (which is given in _radial_ distance) to cover the diagonals of the plotted area (plus 2.5% to remove any fuzziness at corners of image)
            whereOutsideBox = np.abs(snap.data["pos"]) > boxmax*np.sqrt(2.0)*1.025

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereOutsideBox,
                errorString = "Remove Outside Box",
                verbose = DEBUG,
                )

            print(
                f"[@{int(snapNumber)}]: Select stars..."
            )

            whereWind = snap.data["age"] < 0.0

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereWind,
                errorString = "Remove Wind from Gas",
                verbose = DEBUG,
                )

            box = [boxmax, boxmax, boxmax]

            # Calculate New Parameters and Load into memory others we want to track
            snap = tr.calculate_tracked_parameters(
                snap,
                oc.elements,
                oc.elements_Z,
                oc.elements_mass,
                oc.elements_solar,
                oc.Zsolar,
                oc.omegabaryon0,
                snapNumber,
                logParameters=HYPARAMS["logParameters"],
                paramsOfInterest=HYPARAMS["saveParams"],
                mappingBool=True,
                box=box,
                numthreads=HYPARAMS['numthreads'],
                DataSavepath=savePathBaseFigureData,
                verbose = DEBUG,
            )

            # Redshift
            redshift = snap.redshift  # z
            aConst = 1.0 / (1.0 + redshift)  # [/]

            # Get lookback time in Gyrs
            # [0] to remove from numpy array for purposes of plot title
            lookback = snap.cosmology_get_lookback_time_from_a(np.array([aConst]))[
                0
            ]  # [Gyrs]


            # snap.data["R"] = snap.data["R"]/rvir

            print(
                f"[@{int(snapNumber)}]: Ages: get_lookback_time_from_a() ..."
            )

            ages = snap.cosmology_get_lookback_time_from_a(snap.data["age"],is_flat=True)

            snap.data["age"] = ages


            print(
                f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            )

            whereSatellite = np.isin(snap.data["subhalo"],np.array([-1,int(HYPARAMS["HaloID"]),np.nan]))==False

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereSatellite,
                errorString = "Remove Satellites",
                verbose = DEBUG,
                )

            # ## Check that radii are still being stored in units of Rvir...
            # if np.all(snap.data["R"][np.where(np.linalg.norm(snap.data["pos"],axis=1)<=HYPARAMS["Router"]*rvir)[0]]<=HYPARAMS["Router"]): 
            #     pass
            # else:
            #     ## if radii are not in units of rvir, set that now...
            #     snap.data["R"] = snap.data["R"]/rvir

            print(
                f"[@{int(snapNumber)}]: Remove beyond {HYPARAMS['Router']:2.2f} x Rvir..."
            )

            whereBeyondVirial = snap.data["R"] > float(HYPARAMS['Router'])*rvir

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereBeyondVirial,
                errorString = f"Remove Beyond {HYPARAMS['Router']:2.2f} x Rvir",
                verbose = DEBUG,
                )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})


            whereNotStars = out["type"]!=4

            out = cr.remove_selection(
                out,
                removalConditionMask = whereNotStars,
                errorString = "Remove all types other than Stars",
                verbose = DEBUG
                )

            selectKey = (baseResLevel, auHalo, resLabel, "Stars", snapNumber)

            out["Redshift"] = np.array([redshift])
            out["Lookback"] = np.array([lookback])
            out["Snap"] = np.array([snapNumber])
            out["Rvir"] = np.array([rvir])
            out["Rdisc"] = np.array([rdisc])

            innerStarsDict = {copy.deepcopy(selectKey) : copy.deepcopy(out)}
            del out


            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            selectKey = (baseResLevel, auHalo, resLabel, snapNumber)

            out["Redshift"] = np.array([redshift])
            out["Lookback"] = np.array([lookback])
            out["Snap"] = np.array([snapNumber])
            out["Rvir"] = np.array([rvir])
            out["Rdisc"] = np.array([rdisc])

            innerFullDataDict = {copy.deepcopy(selectKey) : copy.deepcopy(out)}
            del out

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            del snap

            whereNotGas= (out["type"]!=0)

            out = cr.remove_selection(
                out,
                removalConditionMask = whereNotGas,
                errorString = "Remove all types other than gas ",
                verbose = DEBUG
                )
            
            # whereNotCGM = (out["R"]<30.0)
            # out = cr.remove_selection(
            #     out,
            #     removalConditionMask = whereNotCGM,
            #     errorString = "Remove all gas within 30.0 kpc",
            #     verbose = DEBUG
            #     )
            
            whereAboveCritDens = (out["ndens"] >= 1.1e-1)

            out = cr.remove_selection(
                out,
                removalConditionMask = whereAboveCritDens,
                errorString = f"Remove ISM gas",
                verbose = DEBUG,
                )
            
            selectKey = (baseResLevel, auHalo, resLabel, snapNumber)

            out["Redshift"] = np.array([redshift])
            out["Lookback"] = np.array([lookback])
            out["Snap"] = np.array([snapNumber])
            out["Rvir"] = np.array([rvir])
            out["Rdisc"] = np.array([rdisc])

            innerCGMgasDict = {copy.deepcopy(selectKey) : copy.deepcopy(out)}
            del out

        
            for key, val in innerStarsDict.items():
                if (int(key[-1]) == int(snapRange[-1])):
                    lastSnapStarsDict.update({key: copy.deepcopy(val)})

            typesCombos ={
                "M200c" : [0,1,4,5],
                "Mstars" : [4],
                "MBH" : [5]
                }

            for label in ["all","ism","cgm"]:
                typesCombos.update({label : [0]})

            print(
                f"[@{snapNumber}]: Halo masses by type..."
            )

            flattened = {}
            for sKey, snapDat in innerFullDataDict.items():
                toCombine = {}
                for label, typeCombo in typesCombos.items():
                    # dataDict starsDict fullDataDict colDict 
                    tmpDat = copy.deepcopy(snapDat)

                    whereNotType = np.isin(tmpDat["type"],np.array(typeCombo))==False
                    tmpDat = cr.remove_selection(
                        tmpDat,
                        removalConditionMask = whereNotType,
                        errorString = f"Remove types from {label} summary data",
                        hush = not DEBUG,
                        verbose = DEBUG,
                    )

                    if label == "ism":
                        whereNotCGM = tmpDat["R"] > HYPARAMS["Router"] * snapDat["Rvir"]
                        tmpDat = cr.remove_selection(
                            tmpDat,
                            removalConditionMask = whereNotCGM,
                            errorString = f"Remove whereNotCGM from {label} summary data",
                            hush = not DEBUG,
                            verbose = DEBUG,
                        )

                        whereBelowCritDens = (tmpDat["ndens"] < 1.1e-1)

                        tmpDat = cr.remove_selection(
                            tmpDat,
                            removalConditionMask = whereBelowCritDens,
                            errorString = f"Remove whereBelowCritDens from {label} summary data",
                            verbose = DEBUG,
                            )
                        label = "M_Gas_"+label
                    elif label == "cgm":

                        whereNotCGM = tmpDat["R"] > HYPARAMS["Router"] * snapDat["Rvir"]
                        tmpDat = cr.remove_selection(
                            tmpDat,
                            removalConditionMask = whereNotCGM,
                            errorString = f"Remove whereNotCGM from {label} summary data",
                            hush = not DEBUG,
                            verbose = DEBUG,
                        )

                        whereAboveCritDens = (tmpDat["ndens"] >= 1.1e-1)

                        tmpDat = cr.remove_selection(
                            tmpDat,
                            removalConditionMask = whereAboveCritDens,
                            errorString = f"Remove whereAboveCritDens from {label} summary data",
                            verbose = DEBUG,
                            )
                        label = "M_Gas_"+label
                    elif label == "all":
                        label = "M_Gas_"+label

                    toCombine.update({label : np.sum(tmpDat["mass"],axis=0)})
                
                # innerFlattened = cr.cr_flatten_wrt_time(toCombine, stack = True, verbose = DEBUG, hush = not DEBUG)
                flattened.update({sKey: copy.deepcopy(toCombine)})

            # unaveragedFlattenedMasses = copy.deepcopy(flattened)
            # if (len(snapRange)>1):
            #     for label, data in flattened.items():
            #         dataCopy = copy.deepcopy(data)
            #         for key,value in data.items():
            #             dataCopy.update({key: np.nanmedian(value,axis=-1)})
            #         flattened[label].update(dataCopy)

            masses = copy.deepcopy(flattened)

            print(
                f"[@{snapNumber}]: CGM summary..."
            )

            flattened = {}
            for sKey, snapDat in innerCGMgasDict.items():
                toCombine = {}
                # dataDict starsDict fullDataDict colDict 
                tmpDat = copy.deepcopy(snapDat)

                toCombine.update({"Ncells" : np.shape(tmpDat["type"][np.where(tmpDat["type"]==0)[0]])[0]})
                toCombine.update({"MedianCellVol" : np.nanmedian(tmpDat["vol"],axis=-1)})
                toCombine.update({"MedianCellMass" : np.nanmedian(tmpDat["mass"],axis=-1)})
                toCombine.update({"MinCellVol" : np.nanmin(tmpDat["vol"],axis=-1)})
                toCombine.update({"MinCellMass" : np.nanmin(tmpDat["mass"],axis=-1)})
                toCombine.update({"MaxCellVol" : np.nanmax(tmpDat["vol"],axis=-1)})
                toCombine.update({"MaxCellMass" : np.nanmax(tmpDat["mass"],axis=-1)})

                # innerFlattened = cr.cr_flatten_wrt_time(toCombine, stack = True, verbose = DEBUG, hush = not DEBUG)
                flattened.update({sKey: copy.deepcopy(toCombine)})


            resolutionStats = copy.deepcopy(flattened)

            print(
                f"[@{snapNumber}]: Hydrogen species masses..."
            )
            
            
            flattened = {}
            for sKey, snapDat in innerFullDataDict.items():
                toCombine = {}
                for label in ["H","HI"]:
                    # dataDict starsDict fullDataDict colDict 
                    tmpDat = copy.deepcopy(snapDat)

                    whereNotGas = tmpDat["type"]!=0
                    tmpDat = cr.remove_selection(
                        tmpDat,
                        removalConditionMask = whereNotGas,
                        errorString = f"Remove whereNotGas from {label} summary data",
                        hush = not DEBUG,
                        verbose = DEBUG,
                    )

                    whereNotCGM = tmpDat["R"] > HYPARAMS["Router"]*rvir
                    tmpDat = cr.remove_selection(
                        tmpDat,
                        removalConditionMask = whereNotCGM,
                        errorString = f"Remove whereNotCGM from {label} summary data",
                        hush = not DEBUG,
                        verbose = DEBUG,
                    )

                    whereAboveCritDens = (tmpDat["ndens"] >= 1.1e-1)

                    tmpDat = cr.remove_selection(
                        tmpDat,
                        removalConditionMask = whereAboveCritDens,
                        errorString = f"Remove whereAboveCritDens from {label} summary data",
                        verbose = DEBUG,
                        )


                    mass = (tmpDat["n_"+label]*((c.parsec * 1e3) ** 3)*tmpDat["vol"]*c.amu*tmpDat["gmet"][:,0]/(c.msol))
                    toCombine.update({"M_"+label+";CGM" : np.sum(mass,axis=0)})
                
                # innerFlattened = cr.cr_flatten_wrt_time(toCombine, stack = True, verbose = DEBUG, hush = not DEBUG)
                flattened.update({sKey: copy.deepcopy(toCombine)})

            # unaveragedFlattenedHydrogenMasses = copy.deepcopy(flattened)
            # if (len(snapRange)>1):
            #     for label, data in flattened.items():
            #         dataCopy = copy.deepcopy(data)
            #         for key,value in data.items():
            #             dataCopy.update({key: np.nanmedian(value,axis=-1)})
            #         flattened[label].update(dataCopy)

            hydrogenMasses = copy.deepcopy(flattened)

            print(
                f"[@{snapNumber}]: Radii..."
            )
            
            flattened = {}
            for sKey, snapDat in innerFullDataDict.items():      
                toCombine = {}
                for label in ["Rdisc","Rvir"]:
                    tmpDat = copy.deepcopy(snapDat)
                    toCombine.update({label : tmpDat[label][0]})
                # innerFlattened = cr.cr_flatten_wrt_time(toCombine, stack = True, verbose = DEBUG, hush = not DEBUG)
                flattened.update({sKey: copy.deepcopy(toCombine)})

            # unaveragedFlattenedRadii = copy.deepcopy(flattened)
            # if (len(snapRange)>1):
            #     for label, data in flattened.items():
            #         dataCopy = copy.deepcopy(data)
            #         for key,value in data.items():
            #             dataCopy.update({key: np.nanmedian(value,axis=-1)})
            #         flattened[label].update(dataCopy)

            radii = copy.deepcopy(flattened)

            print(
                f"[@{snapNumber}]: Combine data..."
            )
            

            inner = {}
            for dat in [masses,resolutionStats,hydrogenMasses,radii]:
                for key, dd in dat.items():
                    for kk, vv in dd.items():
                        inner[kk] = copy.deepcopy(np.asarray(vv))

            selectKey = (baseResLevel, auHalo, resLabel, snapNumber)

            out = {selectKey : copy.deepcopy(inner)}

            dataDict.update(out)
            
            starsDict.update(innerStarsDict)
            # fullDataDict.update(innerFullDataDict)
            # if len(HYPARAMS["colParams"])>0:
                # colDict.update(innerColDict)
            print(
                f"[@{snapNumber}]: Snap calculation complete!"
            )

        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
        #   Quantitative Time-Averaged Summary
        # =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        selectKey = (baseResLevel, auHalo, resLabel)
        print("\n"+f"Starting {selectKey} quantitative time-averaged summary ...")

        HYPARAMS.update({'halo': auHalo})
        selectKey = (baseResLevel, auHalo, resLabel)

        # innerStatsDict = {}
        # innerColStatsDict = {}


        selectKey = (baseResLevel, auHalo, resLabel)

        flattened = cr.cr_flatten_wrt_time(dataDict, stack = True, verbose = DEBUG, hush = not DEBUG)
        if (len(snapRange)>1):
            for sKey, data in flattened.items():
                dataCopy = copy.deepcopy(data)
                for label,value in data.items():
                    dataCopy.update({label: np.asarray([np.nanmedian(value,axis=-1)])})
                flattened[sKey].update(dataCopy)

        simOut = copy.deepcopy(flattened)
        # for key, dat in flattened.items():
        #     inner = copy.deepcopy(dat)
        #     res, crAlfven = key

        #     alfvenindicator = "" 
        #     crindicator = ""
        #     crAlfvenSplit = crAlfven.split("_")
        #     if crAlfvenSplit[-1] == "Alfven":
        #         alfvenindicator = "_".join(crAlfvenSplit[-2:])
        #     else:
        #         alfvenindicator = ""
        #     crindicator = "_".join(crAlfvenSplit[:2])

        if HYPARAMS["SFR"] is True:
            selectKeyLast = (baseResLevel, auHalo, resLabel,
                    "Stars",
                    snapRange[-1]
                    )

            if HYPARAMS["ageWindow"] is None:
                selectKeyFirst = (baseResLevel, auHalo, resLabel,
                        "Stars",
                        snapRange[0]
                        )
                HYPARAMS["ageWindow"] = starsDict[selectKeyFirst]["Lookback"][0]
                                                            

            sfrData = copy.deepcopy(lastSnapStarsDict[selectKeyLast])

            analysisParam = "age"
            weightKey = "gima"

            for excl in singleValueKeys:
                if excl in list(sfrData.keys()):
                    sfrData.pop(excl)

            plotData = sfrData[analysisParam]

            whereAgeBelowLimit = np.full(shape=np.shape(plotData),fill_value=True)
            if HYPARAMS["ageWindow"] is not None:
                print("Minimum age detected = ", np.nanmin(plotData), "Gyr")
                # minAge = np.nanmin(tmpPlot) + ((np.nanmax(tmpPlot) - np.nanmin(tmpPlot))*ageWindow)
                maxAge = np.nanmin(plotData)+HYPARAMS["ageWindow"]
                print("Maximum age for plotting = ", maxAge, "Gyr")

                whereAgeBelowLimit = plotData<=maxAge
                print("Number of data points meeting age = ",np.shape(np.where(whereAgeBelowLimit==True)[0])[0])
                whereAgeBeyondLimit = plotData>maxAge
                sfrData = cr.remove_selection(
                    sfrData,
                    removalConditionMask = whereAgeBeyondLimit,
                    errorString = "Remove stars formed beyond age limit",
                    hush = not DEBUG,
                    verbose = DEBUG,
                )

            plotData = sfrData[analysisParam]
            if analysisParam in HYPARAMS["logParameters"]:
                plotData = np.log10(plotData).copy()
            else:
                plotData = plotData.copy()
            print(
            f"[@{snapNumber}]: SFR Calculation..."
            )

            # Calculate SFR
            cumulativeStellarMass = np.sum(sfrData[weightKey],axis=0)
            delta = HYPARAMS["ageWindow"]
            sfrval = cumulativeStellarMass/(delta*1e9) # SFR [per yr]


            selectKey = (baseResLevel, auHalo, resLabel)


            simOut[selectKey]["SFR"] = np.asarray([copy.deepcopy(sfrval)])

        finalOut.update({selectKey : copy.deepcopy(simOut)})

        print(
            f"[@{snapNumber}]: Sim calculations complete!"
        )

        sheet = "_".join(list(selectKey))
        filename = f"HY-Data_{sheet}.xlsx"
        excel = pd.ExcelWriter(path=FigureSavepathBase+filename,mode="w")
        with excel as writer:
            for (selectKey, simDict) in finalOut.items():
                df = pd.DataFrame.from_dict(simDict[selectKey])
                df.to_excel(writer)

        print(
            f"[@{HYPARAMS['halo']}]: Finished halo..."
        )
    print(
        "\n"+
        f"Finished completely! :)"
    )
