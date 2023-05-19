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

HYPARAMSPATH = "HYParams.json"
HYPARAMS = json.load(open(HYPARAMSPATH, "r"))

if HYPARAMS["coldenspixreslos"] is None:
    HYPARAMS["coldenspixreslos"] = (HYPARAMS["coldenslos"]/HYPARAMS["boxlos"])*HYPARAMS["pixreslosproj"]

if "mass" not in HYPARAMS["colParams"]:
    HYPARAMS["colParams"]+=["mass"]

if HYPARAMS["ageWindow"] is not None:
    HYPARAMS["SFRBins"] = int(math.floor(HYPARAMS["ageWindow"]/HYPARAMS["windowBins"]))
else:
    HYPARAMS["SFRBins"]  = HYPARAMS["Nbins"] 

loadPathBase = "/home/cosmos/"
loadDirectories = [
    "spxfv/Auriga/level4_cgm/h5_standard",
    "spxfv/auriga/level4_cgm/h5_1kpc",
    "c1838736/auriga/level4_cgm/h5_1kpc-hy-500pc",
    "c1838736/auriga/spxfv/auriga/level4_cgm/h5_500pc",
    "c1838736/auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition",
    "c1838736/auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition",
    "c1838736/auriga/level4_cgm/h5_500pc-hy-250pc",
    "c1838736/auriga/level3_cgm_almost/h5_standard",
    #"h5_standard",
    #"h5_2kpc",
    #"h5_1kpc",
    #"snapshot-restart-of-2kpc/h5_1kpc_snapshot-restart-of-2kpc",
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
    # "snapshot-restart-of-2kpc/no-self-shielding/h5_1kpc_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/no-self-shielding/h5_hy-v2_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-1kpc/h5_hy-v2_snapshot-restart-of-1kpc",
    ]

simulations = []
savePaths = []

for dir in loadDirectories:
    loadpath = loadPathBase+dir+"/output/"
    simulations.append(loadpath)
    savepath = "./" + dir + "/"
    savePaths.append(savepath)



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

xlimDict = {
    "R": {"xmin": 0.0, "xmax": 175.0},
    "mass": {"xmin": 4.0, "xmax": 9.0},
    "L": {"xmin": 3.0, "xmax": 4.5},
    "T": {"xmin": 3.75, "xmax": 7.0},
    "n_H": {},#{"xmin": -5.5, "xmax": -0.5},
    "n_H_col": {},#{"xmin": 19.0, "xmax": 21.5},
    "n_HI" :{},
    "n_HI_col" : {},#{"xmin": 14.0, "xmax": 21.0},
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
    "vol": {},#{"xmin": 0.5**4, "xmax": 4.0**4}
    "cool_length" : {},#{"xmin": -1.0 "xmax": 3.0},
    "csound" : {},#
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
    for (loadpath,savePathBase) in zip(simulations,savePaths):
        print(loadpath)
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
                loadonlytype=[0,1,4],#[0, 1, 2, 3, 4, 5],
                lazy_load=False,
                subfind=snap_subfind,
                loadonlyhalo=int(HYPARAMS["HaloID"]),
            )

            print(f"[@{int(snapNumber)}]: Rotate and centre snapshot")
            snap.calc_sf_indizes(snap_subfind, halolist=[int(HYPARAMS["HaloID"])])
            if rotation_matrix is None:
                print(f"[@{int(snapNumber)}]: New rotation of snapshots")
                rotation_matrix = snap.select_halo(snap_subfind, do_rotation=True)
            else:
                print(f"[@{int(snapNumber)}]: Existing rotation of snapshots")
                snap.select_halo(snap_subfind, do_rotation=False)
                snap.rotateto(
                    rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
                )


            print(
                f"[@{int(snapNumber)}]: SnapShot loaded at RedShift z={snap.redshift:0.05e}"
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

            print(
                f"[@{int(snapNumber)}]: Select stars..."
            )

            whereWind = snap.data["age"] < 0.0

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereWind,
                errorString = "Remove Wind from Gas",
                verbose = verbose,
                )


            Rvir = (snap_subfind.data["frc2"] * 1e3)[int(0)]

            rmax = 175.0
            boxmax = rmax
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
                verbose = False,
            )

            print(
                f"[@{int(snapNumber)}]: Remove beyond {HYPARAMS['rvirFrac']:2.2f} x Virial Radius..."
            )

            whereOutsideVirial = snap.data["R"] > Rvir*HYPARAMS["rvirFrac"]#*1.5

            xlimDict["R"]["xmax"] = Rvir*HYPARAMS["rvirFrac"]

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereOutsideVirial,
                errorString = "Remove Outside Virial",
                verbose = verbose,
                )

            print(
                f"[@{int(snapNumber)}]: Ages: get_lookback_time_from_a() ..."
            )

            ages = snap.cosmology_get_lookback_time_from_a(snap.data["age"],is_flat=True)

            snap.data["age"] = ages


            whereOthers = np.isin(snap.data["type"],np.array([1,2,3,5]))

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereOthers,
                errorString = "Remove all types other than Gas and Stars",
                verbose = verbose
                )


            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            )

            whereSatellite = np.isin(out["subhalo"],np.array([-1,int(HYPARAMS["HaloID"]),np.nan]))==False

            out = cr.remove_selection(
                out,
                removalConditionMask = whereSatellite,
                errorString = "Remove Satellites",
                verbose = verbose,
                )

            print(
                f"[@{int(snapNumber)}]: PDF of mass vs R plot..."
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["R"],
                savePathBase = savePathBase,
                saveCurve = True,
                forceLogMass = HYPARAMS["forceLogMass"],
            )

            print(
                f"[@{int(snapNumber)}]: Cumulative PDF of mass vs R plot..."
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["R"],
                cumulative = True,
                savePathBase = savePathBase,
                saveCurve = True,
                forceLogMass = HYPARAMS["forceLogMass"],

            )

            print(
                f"[@{int(snapNumber)}]: Normalised Cumulative PDF of mass vs R plot..."
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["R"],
                cumulative = True,
                normalise = True,
                savePathBase = savePathBase,
                saveCurve = True,
                forceLogMass = HYPARAMS["forceLogMass"],

            )

            #print(
            #    f"[@{int(snapNumber)}]: By Type PDF of mass vs R plot..."
            #)

            #apt.pdf_versus_plot(
            #    out,
            #    ylabel,
            #    xlimDict,
            #    HYPARAMS["logParameters"],
            #    snapNumber,
            #    weightKeys = ['mass'],
            #    xParams = ["R"],
            #    savePathBase = savePathBase,
            #    saveCurve = True,
            #    byType = True,
            #    forceLogMass = HYPARAMS["forceLogMass"],
            #)

            #print(
            #    f"[@{int(snapNumber)}]: By Type Cumulative PDF of mass vs R plot..."
            #)

            #apt.pdf_versus_plot(
            #    out,
            #    ylabel,
            #    xlimDict,
            #    HYPARAMS["logParameters"],
            #    snapNumber,
            #    weightKeys = ['mass'],
            #    xParams = ["R"],
            #    cumulative = True,
            #    savePathBase = savePathBase,
            #    saveCurve = True,
            #    byType = True,
            #    forceLogMass = HYPARAMS["forceLogMass"],
            #)

            #print(
            #    f"[@{int(snapNumber)}]: By Type Normalised Cumulative PDF of mass vs R plot..."
            #)

            #apt.pdf_versus_plot(
            #    out,
            #    ylabel,
            #    xlimDict,
            #    HYPARAMS["logParameters"],
            #    snapNumber,
            #    weightKeys = ['mass'],
            #    xParams = ["R"],
            #    cumulative = True,
            #    normalise = True,
            #    savePathBase = savePathBase,
            #    saveCurve = True,
            #    byType = True,
            #    forceLogMass = HYPARAMS["forceLogMass"],
            #)

            print(
                f"[@{int(snapNumber)}]: Remove all types other than Gas and Stars..."
            )

            whereOthers = np.isin(snap.data["type"],np.array([1,2,3,5]))

            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereOthers,
                errorString = "Remove all types other than Gas and Stars",
                verbose = verbose
                )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            )

            whereSatellite = np.isin(out["subhalo"],np.array([-1,int(HYPARAMS["HaloID"]),np.nan]))==False

            out = cr.remove_selection(
                out,
                removalConditionMask = whereSatellite,
                errorString = "Remove Satellites",
                verbose = verbose,
                )

            print(
                f"[@{int(snapNumber)}]: SFR plot..."
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['gima'],
                xParams = ["age"],
                ageWindow = HYPARAMS["ageWindow"],
                Nbins = HYPARAMS["SFRBins"],
                savePathBase = savePathBase,
                saveCurve = True,
                SFR = True,
            )

            print(
                f"[@{int(snapNumber)}]: Cumulative SFR plot..."
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['gima'],
                xParams = ["age"],
                ageWindow = HYPARAMS["ageWindow"],
                Nbins = HYPARAMS["SFRBins"],
                savePathBase = savePathBase,
                cumulative = True,
                saveCurve = True,
                SFR = True,
            )


            print(
                f"[@{int(snapNumber)}]: Normalised Cumulative SFR plot..."
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['gima'],
                xParams = ["age"],
                ageWindow = HYPARAMS["ageWindow"],
                Nbins = HYPARAMS["SFRBins"],
                savePathBase = savePathBase,
                cumulative = True,
                normalise = True,
                saveCurve = True,
                SFR = True,
            )

            print(
                f"[@{int(snapNumber)}]: Remove stars..."
            )
            whereStars = snap.data["type"] == 4
            snap = cr.remove_selection(
                snap,
                removalConditionMask = whereStars,
                errorString = "Remove Stars from Gas",
                verbose = verbose
                )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            )

            whereSatellite = np.isin(out["subhalo"],np.array([-1,int(HYPARAMS["HaloID"]),np.nan]))==False

            out = cr.remove_selection(
                out,
                removalConditionMask = whereSatellite,
                errorString = "Remove Satellites",
                verbose = verbose,
                )

            colout = {}
            for param in HYPARAMS["colParams"]:
                paramSplitList = param.split("_")

                #if paramSplitList[-1] == "col":

                print(
                    f"[@{int(snapNumber)}]: Calculate {param}..."
                )
                boxsize = Rvir*HYPARAMS["rvirFracImages"]*2.0

                tmpdict = apt.plot_slices(snap,
                    ylabel=ylabel,
                    xlimDict=xlimDict,
                    logParameters = HYPARAMS["logParameters"],
                    snapNumber=snapNumber,
                    sliceParam = param,
                    Axes=HYPARAMS["Axes"],
                    xsize = HYPARAMS["xsizeImages"],
                    ysize = HYPARAMS["ysizeImages"],
                    colourmapMain=HYPARAMS["colourmapMain"],
                    boxsize=boxsize,
                    boxlos=HYPARAMS["coldenslos"],
                    pixreslos=HYPARAMS["coldenspixreslos"],
                    pixres=HYPARAMS["pixresproj"],
                    projection = True,
                    DPI = HYPARAMS["DPI"],
                    numthreads=HYPARAMS["numthreads"],
                    savePathBase = savePathBase,
                    saveFigure = False,
                )

                KpcTocm = 1e3 * c.parsec
                convert = float(HYPARAMS["coldenspixreslos"])*KpcTocm
                if paramSplitList[-1] == "col":
                    nonColParam = "_".join(paramSplitList[:-1])
                else:
                    nonColParam = param
                nonColShape = snap.data[nonColParam].shape
                if snap.data[nonColParam].ndim>1:
                    newShape=(-1,nonColShape[-1])
                else:
                    newShape = (-1)
                if paramSplitList[-1] == "col":
                    colout.update({param: (copy.deepcopy(tmpdict[param]["grid"])*convert).reshape(newShape)})
                    #snap.data[param] = 
                else:
                    colout.update({param: (copy.deepcopy(tmpdict[param]["grid"])).reshape(newShape)})
                    #snap.data[param] = 

                #colout.update({param: copy.deepcopy(snap.data[param])})
                #snap.data[param+"x"] = 
                #snap.data[param+"y"] = 
                #colout.update({param+"x": (copy.deepcopy(tmpdict[param]["x"])).reshape(newShape)})
                #colout.update({param+"y": (copy.deepcopy(tmpdict[param]["y"])).reshape(newShape)})
                if HYPARAMS["xParam"] is not "R" : 
                    raise Exception("FAILURE! Col Density plotting not adapted for xParams other than R!"
                                                                   +"\n"
                                                                   +"Please add in functionality and edit/remove this message!")
                if HYPARAMS["xParam"] not in list(colout.keys()):
                    xx = (copy.deepcopy(tmpdict[param]["x"])).reshape(newShape)
                    xx = np.array(
                        [
                            (x1 + x2) / 2.0
                            for (x1, x2) in zip(xx[:-1], xx[1:])
                        ]
                    )
                    yy = (copy.deepcopy(tmpdict[param]["y"])).reshape(newShape)
                    yy = np.array(
                        [
                            (x1 + x2) / 2.0
                            for (x1, x2) in zip(yy[:-1], yy[1:])
                        ]
                    )
                    #key = xParam+"_col"
                    values = np.linalg.norm(np.asarray(np.meshgrid(xx,yy)).reshape(-1,2), axis=1)
                    #snap.data["R_col"] = values
                    colout.update({HYPARAMS["xParam"]: values})
                    colout.update({"type": np.full(shape=values.shape, fill_value=0)})



            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snap.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: PDF of gas (mass vs T or vol) plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["pdfParams"],
                savePathBase = savePathBase,
                saveCurve = True,
                forceLogMass = HYPARAMS["forceLogMass"],
            )

            apt.pdf_versus_plot(
                dataDict = colout,
                ylabel = ylabel,
                xlimDict = xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber = snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["colParams"],
                savePathBase = savePathBase,
                saveCurve = True,
                forceLogMass = HYPARAMS["forceLogMass"],
            )

            print(
                f"[@{int(snapNumber)}]: Cumulative PDF of gas (mass vs T or vol) plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["pdfParams"],
                savePathBase = savePathBase,
                cumulative = True,
                saveCurve = True,
                forceLogMass = HYPARAMS["forceLogMass"],
            )

            apt.pdf_versus_plot(
                dataDict = colout,
                ylabel = ylabel,
                xlimDict = xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber = snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["colParams"],
                savePathBase = savePathBase,
                cumulative = True,
                saveCurve = True,
                forceLogMass = HYPARAMS["forceLogMass"],
            )

            print(
                f"[@{int(snapNumber)}]: Normalised Cumulative PDF of gas (mass vs T or vol) plot"
            )

            apt.pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["pdfParams"],
                savePathBase = savePathBase,
                cumulative = True,
                normalise = True,
                saveCurve = True,
                forceLogMass = HYPARAMS["forceLogMass"],
            )

            apt.pdf_versus_plot(
                dataDict = colout,
                ylabel = ylabel,
                xlimDict = xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber = snapNumber,
                weightKeys = ['mass'],
                xParams = HYPARAMS["colParams"],
                savePathBase = savePathBase,
                cumulative = True,
                normalise = True,
                saveCurve = True,
                forceLogMass = HYPARAMS["forceLogMass"],
            )

            print(
                f"[@{int(snapNumber)}]: Slice plot"
            )

            apt.hy_plot_slices(snap,
                snapNumber,
                xsize = 15.00,
                ysize=7.50,
                pixres=HYPARAMS["pixres"],
                DPI = HYPARAMS["DPI"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Slice plot Quad"
            )

            apt.hy_plot_slices_quad(snap,
                snapNumber,
                xsize = 15.00,
                ysize=15.00,
                pixres=HYPARAMS["pixres"],
                DPI = HYPARAMS["DPI"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Projection plot"
            )

            apt.hy_plot_projections(snap,
                snapNumber,
                xsize = 15.00,
                ysize=7.50,
                boxlos=HYPARAMS["boxlos"],
                pixreslos=HYPARAMS["pixreslosproj"],
                pixres=HYPARAMS["pixresproj"],
                DPI = HYPARAMS["DPI"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )


            print(
                f"[@{int(snapNumber)}]: Generalised T Projection plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "T",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["boxlos"],
                pixreslos=HYPARAMS["pixreslosproj"],
                pixres=HYPARAMS["pixresproj"],
                projection = True,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Generalised T Slice plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "T",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["boxlos"],
                pixreslos=HYPARAMS["pixreslos"],
                pixres=HYPARAMS["pixres"],
                projection = False,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Generalised n_H Projection plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "n_H",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["boxlos"],
                pixreslos=HYPARAMS["pixreslosproj"],
                pixres=HYPARAMS["pixresproj"],
                projection = True,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Generalised n_H Slice plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "n_H",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["boxlos"],
                pixreslos=HYPARAMS["pixreslos"],
                pixres=HYPARAMS["pixres"],
                projection = False,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )
    

            print(
                f"[@{int(snapNumber)}]: Generalised n_H_col Projection plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "n_H_col",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["coldenslos"],
                pixreslos=HYPARAMS["coldenspixreslos"],
                pixres=HYPARAMS["pixresproj"],
                projection = True,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Generalised n_H_col Slice plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "n_H_col",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["coldenslos"],
                pixreslos=HYPARAMS["coldenspixreslos"],
                pixres=HYPARAMS["pixres"],
                projection = False,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )


            print(
                f"[@{int(snapNumber)}]: Generalised n_HI Projection plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "n_HI",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["boxlos"],
                pixreslos=HYPARAMS["pixreslosproj"],
                pixres=HYPARAMS["pixresproj"],
                projection = True,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Generalised n_HI Slice plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "n_HI",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["boxlos"],
                pixreslos=HYPARAMS["pixreslos"],
                pixres=HYPARAMS["pixres"],
                projection = False,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Generalised n_HI_col Projection plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "n_HI_col",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["coldenslos"],
                pixreslos=HYPARAMS["coldenspixreslos"],
                pixres=HYPARAMS["pixresproj"],
                projection = True,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Generalised n_HI_col Slice plot"
            )

            _ = apt.plot_slices(snap,
                ylabel=ylabel,
                xlimDict=xlimDict,
                logParameters = HYPARAMS["logParameters"],
                snapNumber=snapNumber,
                sliceParam = "n_HI_col",
                Axes=HYPARAMS["Axes"],
                xsize = HYPARAMS["xsizeImages"],
                ysize = HYPARAMS["xsizeImages"],
                colourmapMain=HYPARAMS["colourmapMain"],
                boxsize=Rvir*HYPARAMS["rvirFracImages"]*2.0,
                boxlos=HYPARAMS["coldenslos"],
                pixreslos=HYPARAMS["coldenspixreslos"],
                pixres=HYPARAMS["pixres"],
                projection = False,
                DPI = HYPARAMS["DPI"],
                numthreads=HYPARAMS["numthreads"],
                savePathBase = savePathBase,
            )

            print(
                "\n"+f"[@{int(snapNumber)}]: Calculate full statistics..."
            )

            statsDict = cr.cr_calculate_statistics(
                out,
                CRPARAMS = HYPARAMS,
                xParam=HYPARAMS["xParam"],
                Nbins=HYPARAMS["Nbins"],
                xlimDict=xlimDict,
                printpercent=5.0,
                exclusions = None,
            )

            splitList = loadpath.split("/")
            baseResLevel, haloLabel = splitList[-4:-2]
            haloSplitList = haloLabel.split("_")
            auHalo, resLabel = haloSplitList[0], "_".join(haloSplitList[1:])
            tmp = copy.copy(statsDict)
            statsDict = {(baseResLevel, haloLabel): tmp}

            print(
                "\n"+f"[@{int(snapNumber)}]: Calculate column density statistics..."
            )

            COLHYPARAMS = copy.deepcopy(HYPARAMS)
            COLHYPARAMS['saveParams']+=COLHYPARAMS['colParams']
            COLHYPARAMS["Nbins"] = int(HYPARAMS["Nbins"]**(2/3))
            colstatsDict = cr.cr_calculate_statistics(
                colout,
                CRPARAMS = COLHYPARAMS,
                xParam=COLHYPARAMS["xParam"],
                Nbins=COLHYPARAMS["Nbins"],
                xlimDict=xlimDict,
                printpercent=5.0,
                exclusions = None,
                weightedStatsBool = False,
            )

            tmp = copy.copy(colstatsDict)
            colstatsDict = {(baseResLevel, haloLabel): tmp}

            print(
                "\n"+f"[@{int(snapNumber)}]: Plot column density medians versus {HYPARAMS['xParam']}..."
            )

            apt.medians_versus_plot(
                colstatsDict,
                COLHYPARAMS,
                ylabel=ylabel,
                xlimDict=xlimDict,
                snapNumber=snapNumber,
                yParam=COLHYPARAMS["colParams"],
                xParam=COLHYPARAMS["xParam"],
                titleBool=COLHYPARAMS["titleBool"],
                DPI = COLHYPARAMS["DPI"],
                xsize = COLHYPARAMS["xsize"],
                ysize = COLHYPARAMS["ysize"],
                fontsize = COLHYPARAMS["fontsize"],
                fontsizeTitle = COLHYPARAMS["fontsizeTitle"],
                opacityPercentiles = COLHYPARAMS["opacityPercentiles"],
                colourmapMain = "tab10",
                savePathBase = savePathBase
            )

            print(
                "\n"+f"[@{int(snapNumber)}]: Plot full statistics medians versus {HYPARAMS['xParam']}..."
            )

            apt.medians_versus_plot(
                statsDict,
                HYPARAMS,
                ylabel=ylabel,
                xlimDict=xlimDict,
                snapNumber=snapNumber,
                yParam=HYPARAMS["mediansParams"],
                xParam=HYPARAMS["xParam"],
                titleBool=HYPARAMS["titleBool"],
                DPI = HYPARAMS["DPI"],
                xsize = HYPARAMS["xsize"],
                ysize = HYPARAMS["ysize"],
                fontsize = HYPARAMS["fontsize"],
                fontsizeTitle = HYPARAMS["fontsizeTitle"],
                opacityPercentiles = HYPARAMS["opacityPercentiles"],
                colourmapMain = "tab10",
                savePathBase = savePathBase
            )

            # # print(
            # #     f"[@{int(snapNumber)}]: Remove beyond 1.5 x Virial Radius..."
            # # )
            # #
            # # whereOutsideVirial = snap.data["R"] > Rvir*rvirFrac#*2.0
            # #
            # # xlimDict["R"]["xmax"] = Rvir*rvirFrac#*1.2
            # #
            # # snap = cr.remove_selection(
            # #     snap,
            # #     removalConditionMask = whereOutsideVirial,
            # #     errorString = "Remove Outside Virial from Gas",
            # #     verbose = verbose,
            # #     )
            #
            # # print(
            # #     f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            # # )
            # # # Make normal dictionary form of snap
            # # out = {}
            # # for key, value in snap.data.items():
            # #     if value is not None:
            # #         out.update({key: copy.deepcopy(value)})
            # #
            # #
            # # print(
            # #     f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            # # )
            # #
            # # whereSatellite = np.isin(out["subhalo"],np.array([-1,int(HYPARAMS["HaloID"]),np.nan]))==False
            # #
            # # out = cr.remove_selection(
            # #     out,
            # #     removalConditionMask = whereSatellite,
            # #     errorString = "Remove Satellites",
            # #     verbose = verbose,
            # #     )

            print(
                f"[@{int(snapNumber)}]: Hist_plot_xyz plot"
            )

            apt.hist_plot_xyz(
                out,
                ylabel,
                xlimDict,
                HYPARAMS["logParameters"],
                snapNumber = snapNumber,
                yParams = HYPARAMS["phasesyParams"],
                xParams = HYPARAMS["phasesxParams"],
                weightKeys = HYPARAMS["phasesWeightParams"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Done"
            )
            plt.close("all")
        print("finished sim:", loadpath)
    print("Finished fully! :)")
