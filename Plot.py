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
from Tracers_Subroutines import *
from CR_Subroutines import *
from Plotting_tools import hy_plot_slices,hy_plot_slices_quad,hy_plot_projections,hist_plot_xyz,pdf_versus_plot,plot_slices
import h5py
import json
import copy
import math
import os

ageWindow = None #(Gyr) before current snapshot SFR evaluation
windowBins = 0.100 #(Gyr) size of ageWindow Bins. Ignored if ageWindow is None
Nbins = 250
snapStart = 99
snapEnd = 99#9 #Max = 192 for high-time res
DEBUG = True
forceLogMass = False
DPI = 200
pixres = 0.1
pixreslos = 0.1
pixresproj = 0.2
pixreslosproj = 0.2
numthreads = 18
rvirFrac = 1.20
rvirFracImages = 1.00

loadPathBase = "/home/cosmos/"
loadDirectories = [
    #"c1838736/Auriga/level3_cgm_almost/h5_standard",
    "spxfv/Auriga/level4_cgm/h5_standard",
    #"spxfv/Auriga/level4_cgm/h5_1kpc",
    #"c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc",
    #"c1838736/Auriga/spxfv/Auriga/level4_cgm/h5_500pc",
    #"c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-l3-mass-res-transition",
    #"c1838736/Auriga/level4_cgm/h5_1kpc-hy-500pc-hard-res-transition",
    #"c1838736/Auriga/level4_cgm/h5_500pc-hy-250pc",
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
    "n_H": {},#{"xmin": -5.5, "xmax": -0.5},
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
    "vol": {},#{"xmin": 0.5**4, "xmax": 4.0**4}
    "cool_length" : {},#{"xmin": -1.0 "xmax": 3.0},
    "csound" : {},#
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
                loadonlyhalo=0,
            )

            print(f"[@{int(snapNumber)}]: Rotate and centre snapshot")
            snap.calc_sf_indizes(snap_subfind, halolist=[0])
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

            snap = remove_selection(
                snap,
                removalConditionMask = whereWind,
                errorString = "Remove Wind from Gas",
                DEBUG = DEBUG,
                )


            Rvir = (snap_subfind.data["frc2"] * 1e3)[int(0)]

            rmax = 175.0
            boxmax = rmax
            box = [boxmax, boxmax, boxmax]

            # Calculate New Parameters and Load into memory others we want to track
            snap = calculate_tracked_parameters(
                snap,
                oc.elements,
                oc.elements_Z,
                oc.elements_mass,
                oc.elements_solar,
                oc.Zsolar,
                oc.omegabaryon0,
                snapNumber,
                # logParameters = logParameters,
                paramsOfInterest=["R","T","Tdens","ndens","rho_rhomean","n_H","gz","cool_length"],
                mappingBool=True,
                box=box,
                numthreads=numthreads,
                verbose = False,
            )

            print(
                f"[@{int(snapNumber)}]: Remove beyond {rvirFrac:2.2f} x Virial Radius..."
            )

            whereOutsideVirial = snap.data["R"] > Rvir*rvirFrac#*1.5

            xlimDict["R"]["xmax"] = Rvir*rvirFrac

            snaptmp = remove_selection(
                snap,
                removalConditionMask = whereOutsideVirial,
                errorString = "Remove Outside Virial",
                DEBUG = DEBUG,
                )

            print(
                f"[@{int(snapNumber)}]: Ages: get_lookback_time_from_a() ..."
            )

            ages = snap.cosmology_get_lookback_time_from_a(snap.data["age"],is_flat=True)

            snap.data["age"] = ages


            whereOthers = np.isin(snap.data["type"],np.array([1,2,3,5]))

            snap = remove_selection(
                snap,
                removalConditionMask = whereOthers,
                errorString = "Remove all types other than Gas and Stars",
                DEBUG = DEBUG
                )


            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snaptmp.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            )

            whereSatellite = np.isin(out["subhalo"],np.array([-1,0,np.nan]))==False

            out = remove_selection(
                out,
                removalConditionMask = whereSatellite,
                errorString = "Remove Satellites",
                DEBUG = DEBUG,
                )

            print(
                f"[@{int(snapNumber)}]: PDF of mass vs R plot..."
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["R"],
                savePathBase = savePathBase,
                saveCurve = True,
                forceLogMass = forceLogMass,
            )

            print(
                f"[@{int(snapNumber)}]: Cumulative PDF of mass vs R plot..."
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["R"],
                cumulative = True,
                savePathBase = savePathBase,
                saveCurve = True,
                forceLogMass = forceLogMass,

            )

            print(
                f"[@{int(snapNumber)}]: Normalised Cumulative PDF of mass vs R plot..."
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["R"],
                cumulative = True,
                normalise = True,
                savePathBase = savePathBase,
                saveCurve = True,
                forceLogMass = forceLogMass,

            )

            #print(
            #    f"[@{int(snapNumber)}]: By Type PDF of mass vs R plot..."
            #)

            #pdf_versus_plot(
            #    out,
            #    ylabel,
            #    xlimDict,
            #    logParameters,
            #    snapNumber,
            #    weightKeys = ['mass'],
            #    xParams = ["R"],
            #    savePathBase = savePathBase,
            #    saveCurve = True,
            #    byType = True,
            #    forceLogMass = forceLogMass,
            #)

            #print(
            #    f"[@{int(snapNumber)}]: By Type Cumulative PDF of mass vs R plot..."
            #)

            #pdf_versus_plot(
            #    out,
            #    ylabel,
            #    xlimDict,
            #    logParameters,
            #    snapNumber,
            #    weightKeys = ['mass'],
            #    xParams = ["R"],
            #    cumulative = True,
            #    savePathBase = savePathBase,
            #    saveCurve = True,
            #    byType = True,
            #    forceLogMass = forceLogMass,
            #)

            #print(
            #    f"[@{int(snapNumber)}]: By Type Normalised Cumulative PDF of mass vs R plot..."
            #)

            #pdf_versus_plot(
            #    out,
            #    ylabel,
            #    xlimDict,
            #    logParameters,
            #    snapNumber,
            #    weightKeys = ['mass'],
            #    xParams = ["R"],
            #    cumulative = True,
            #    normalise = True,
            #    savePathBase = savePathBase,
            #    saveCurve = True,
            #    byType = True,
            #    forceLogMass = forceLogMass,
            #)

            print(
                f"[@{int(snapNumber)}]: Remove all types other than Gas and Stars..."
            )

            whereOthers = np.isin(snap.data["type"],np.array([1,2,3,5]))

            snap = remove_selection(
                snap,
                removalConditionMask = whereOthers,
                errorString = "Remove all types other than Gas and Stars",
                DEBUG = DEBUG
                )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snaptmp.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            )

            whereSatellite = np.isin(out["subhalo"],np.array([-1,0,np.nan]))==False

            out = remove_selection(
                out,
                removalConditionMask = whereSatellite,
                errorString = "Remove Satellites",
                DEBUG = DEBUG,
                )

            print(
                f"[@{int(snapNumber)}]: SFR plot..."
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['gima'],
                xParams = ["age"],
                ageWindow = ageWindow,
                Nbins = SFRBins,
                savePathBase = savePathBase,
                saveCurve = True,
                SFR = True,
            )

            print(
                f"[@{int(snapNumber)}]: Cumulative SFR plot..."
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['gima'],
                xParams = ["age"],
                ageWindow = ageWindow,
                Nbins = SFRBins,
                savePathBase = savePathBase,
                cumulative = True,
                saveCurve = True,
                SFR = True,
            )


            print(
                f"[@{int(snapNumber)}]: Normalised Cumulative SFR plot..."
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['gima'],
                xParams = ["age"],
                ageWindow = ageWindow,
                Nbins = SFRBins,
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
            snap = remove_selection(
                snap,
                removalConditionMask = whereStars,
                errorString = "Remove Stars from Gas",
                DEBUG = DEBUG
                )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snaptmp.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})

            print(
                f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            )

            whereSatellite = np.isin(out["subhalo"],np.array([-1,0,np.nan]))==False

            out = remove_selection(
                out,
                removalConditionMask = whereSatellite,
                errorString = "Remove Satellites",
                DEBUG = DEBUG,
                )

            print(
                f"[@{int(snapNumber)}]: PDF of gas (mass vs T or vol) plot"
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["T","vol","n_H"],
                savePathBase = savePathBase,
                saveCurve = True,
                forceLogMass = forceLogMass,
            )

            print(
                f"[@{int(snapNumber)}]: Cumulative PDF of gas (mass vs T or vol) plot"
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["T","vol","n_H"],
                savePathBase = savePathBase,
                cumulative = True,
                saveCurve = True,
                forceLogMass = forceLogMass,
            )

            print(
                f"[@{int(snapNumber)}]: Normalised Cumulative PDF of gas (mass vs T or vol) plot"
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["T","vol","n_H"],
                savePathBase = savePathBase,
                cumulative = True,
                normalise = True,
                saveCurve = True,
                forceLogMass = forceLogMass,
            )


            print(
                f"[@{int(snapNumber)}]: Slice plot"
            )

            hy_plot_slices(snap,
                snapNumber,
                xsize = 15.00,
                ysize=7.50,
                pixres=pixres,
                DPI = DPI,
                boxsize=Rvir*rvirFracImages*2.0,
                numthreads=numthreads,
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Slice plot Quad"
            )

            hy_plot_slices_quad(snap,
                snapNumber,
                xsize = 15.00,
                ysize=15.00,
                pixres=pixres,
                DPI = DPI,
                boxsize=Rvir*rvirFracImages*2.0,
                numthreads=numthreads,
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Projection plot"
            )

            hy_plot_projections(snap,
                snapNumber,
                xsize = 15.00,
                ysize=7.50,
                boxlos=50.0,
                pixreslos=pixreslosproj,
                pixres=pixresproj,
                DPI = DPI,
                boxsize=Rvir*rvirFracImages*2.0,
                numthreads=numthreads,
                savePathBase = savePathBase,
            )

            ##print(
            ##    f"[@{int(snapNumber)}]: Generalised T Projection plot"
            ##)

            ##_ = plot_slices(snap,
            ##    ylabel=ylabel,
            ##    xlimDict=xlimDict,
            ##    logParameters = logParameters,
            ##    snapNumber=snapNumber,
            ##    sliceParam = "T",
            ##    xsize = 7.50,
            ##    ysize = 7.50,
            ##    boxsize=Rvir*rvirFracImages*2.0,
            ##    boxlos=50.0,
            ##    pixreslos=pixreslosproj,
            ##    pixres=pixresproj,
            ##    projection = True,
            ##    DPI = DPI,
            ##    numthreads=numthreads,
            ##    savePathBase = savePathBase,
            ##)

            ##print(
            ##    f"[@{int(snapNumber)}]: Generalised T Slice plot"
            ##)

            ##_ = plot_slices(snap,
            ##    ylabel=ylabel,
            ##    xlimDict=xlimDict,
            ##    logParameters = logParameters,
            ##    snapNumber=snapNumber,
            ##    sliceParam = "T",
            ##    xsize = 7.50,
            ##    ysize = 7.50,
            ##    boxsize=Rvir*rvirFracImages*2.0,
            ##    boxlos=50.0,
            ##    pixreslos=pixreslos,
            ##    pixres=pixres,
            ##    projection = False,
            ##    DPI = DPI,
            ##    numthreads=numthreads,
            ##    savePathBase = savePathBase,
            ##)

            ##print(
            ##    f"[@{int(snapNumber)}]: Generalised Projection plot"
            ##)

            ##_ = plot_slices(snap,
            ##    ylabel=ylabel,
            ##    xlimDict=xlimDict,
            ##    logParameters = logParameters,
            ##    snapNumber=snapNumber,
            ##    sliceParam = "n_H",
            ##    xsize = 7.50,
            ##    ysize = 7.50,
            ##    boxsize=Rvir*rvirFracImages*2.0,
            ##    boxlos=50.0,
            ##    pixreslos=pixreslosproj,
            ##    pixres=pixresproj,
            ##    projection = True,
            ##    DPI = DPI,
            ##    numthreads=numthreads,
            ##    savePathBase = savePathBase,
            ##)

            ##print(
            ##    f"[@{int(snapNumber)}]: Generalised Slice plot"
            ##)

            ##_ = plot_slices(snap,
            ##    ylabel=ylabel,
            ##    xlimDict=xlimDict,
            ##    logParameters = logParameters,
            ##    snapNumber=snapNumber,
            ##    sliceParam = "n_H",
            ##    xsize = 7.50,
            ##    ysize = 7.50,
            ##    boxsize=Rvir*rvirFracImages*2.0,
            ##    boxlos=50.0,
            ##    pixreslos=pixreslos,
            ##    pixres=pixres,
            ##    projection = False,
            ##    DPI = DPI,
            ##    numthreads=numthreads,
            ##    savePathBase = savePathBase,
            ##)

            # # print(
            # #     f"[@{int(snapNumber)}]: Remove beyond 1.5 x Virial Radius..."
            # # )
            # #
            # # whereOutsideVirial = snap.data["R"] > Rvir*rvirFrac#*2.0
            # #
            # # xlimDict["R"]["xmax"] = Rvir*rvirFrac#*1.2
            # #
            # # snaptmp = remove_selection(
            # #     snap,
            # #     removalConditionMask = whereOutsideVirial,
            # #     errorString = "Remove Outside Virial from Gas",
            # #     DEBUG = DEBUG,
            # #     )
            #
            # # print(
            # #     f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            # # )
            # # # Make normal dictionary form of snap
            # # out = {}
            # # for key, value in snaptmp.data.items():
            # #     if value is not None:
            # #         out.update({key: copy.deepcopy(value)})
            # #
            # #
            # # print(
            # #     f"[@{int(snapNumber)}]: Remove other halos from dictionary..."
            # # )
            # #
            # # whereSatellite = np.isin(out["subhalo"],np.array([-1,0,np.nan]))==False
            # #
            # # out = remove_selection(
            # #     out,
            # #     removalConditionMask = whereSatellite,
            # #     errorString = "Remove Satellites",
            # #     DEBUG = DEBUG,
            # #     )

            print(
                f"[@{int(snapNumber)}]: Hist_plot_xyz plot"
            )

            hist_plot_xyz(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber = snapNumber,
                yParams = ["T","ndens"],
                xParams = ["R","rho_rhomean","vol","ndens"],
                weightKeys = ["mass"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Done"
            )
            plt.close("all")
        print("finished sim:", loadpath)
    print("Finished fully! :)")
