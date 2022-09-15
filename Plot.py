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
import h5py
import json
import copy
import math
import os

snapNumber = 104
loadpath = "/home/universe/c1838736/Auriga/level5_cgm/h5_hybrid-dev/output/"
numthreads = 8

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
}

xlimDict = {
    "R": {"xmin": 0.0, "xmax": 175.0},
    # "mass": {"xmin": 5.0, "xmax": 9.0},
    "L": {"xmin": 3.0, "xmax": 4.5},
    "T": {"xmin": 3.75, "xmax": 6.5},
    "n_H": {"xmin": -5.5, "xmax": -0.5},
    "B": {"xmin": -2.5, "xmax": 1.0},
    "vrad": {"xmin": -150.0, "xmax": 150.0},
    "gz": {"xmin": -1.5, "xmax": 0.5},
    "P_thermal": {"xmin": 0.5, "xmax": 3.5},
    "P_CR": {"xmin": -1.5, "xmax": 5.5},
    "PCR_Pthermal": {"xmin": -2.0, "xmax": 2.0},
    "P_magnetic": {"xmin": -2.0, "xmax": 4.5},
    "P_kinetic": {"xmin": 0.0, "xmax": 6.0},
    "P_tot": {"xmin": -1.0, "xmax": 7.0},
    "Pthermal_Pmagnetic": {"xmin": -1.5, "xmax": 3.0},
    "tcool": {"xmin": -3.5, "xmax": 2.0},
    "theat": {"xmin": -4.0, "xmax": 4.0},
    "tff": {"xmin": -1.5, "xmax": 0.75},
    "tcool_tff": {"xmin": -2.5, "xmax": 2.0},
    "rho_rhomean": {"xmin": 1.5, "xmax": 6.0},
    "dens": {"xmin": -30.0, "xmax": -22.0},
    "ndens": {"xmin": -6.0, "xmax": 2.0},
}

logParameters = ["dens","ndens","rho_rhomean","csound","T","n_H","B","gz","L","P_thermal","P_magnetic","P_kinetic","P_tot","Pthermal_Pmagnetic", "P_CR", "PCR_Pthermal","gah","Grad_T","Grad_n_H","Grad_bfld","Grad_P_CR","tcool","theat","tcross","tff","tcool_tff","mass","gima"]


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


def plot_slices(snapGas,
    snapNumber,
    fontsize=13,
    fontsizeTitle=14,
    titleBool=True,
    Axes=[0, 1],
    boxsize=400.0,
    pixres=0.1,
    DPI=200,
    CMAP=None,
    numthreads=10,
):
    savePath = f"./Plots/Slices/"
    tmp = "./"

    for savePathChunk in savePath.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    if CMAP == None:
        cmap = plt.get_cmap("inferno")
    else:
        cmap = CMAP

    # Axes Labels to allow for adaptive axis selection
    AxesLabels = ["x", "y", "z"]

    # Centre image on centre of simulation (typically [0.,0.,0.] for centre of HaloID in set_centre)
    imgcent = [0.0, 0.0, 0.0]

    # --------------------------#
    ## Slices and Projections ##
    # --------------------------#

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    # slice_nH    = snap.get_Aslice("n_H", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    #
    # slice_B   = snap.get_Aslice("B", box = [boxsize,boxsize],\
    #  center = imgcent, nx = int(boxsize/pixres), ny = int(boxsize/pixres),\
    #  axes = Axes, proj = False, numthreads=16)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
    nprojections = 2
    # print(np.unique(snapGas.type))
    print("\n" + f"Projection 1 of {nprojections}")

    slice_T = snapGas.get_Aslice(
        "T",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    print("\n" + f" Projection 2 of {nprojections}")

    slice_vol = snapGas.get_Aslice(
        "vol",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    # ------------------------------------------------------------------------------#
    # PLOTTING TIME
    # Set plot figure sizes
    xsize = 10.0
    ysize = 10.0
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    if titleBool is True:
        # Redshift
        redshift = snapGas.redshift  # z
        aConst = 1.0 / (1.0 + redshift)  # [/]

        # [0] to remove from numpy array for purposes of plot title
        tlookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
            0
        ]  # [Gyrs]
    # ==============================================================================#
    #
    #           Quad Plot for standard video
    #
    # ==============================================================================#
    print(f" Quad Plot...")

    fullTicks = [xx for xx in np.linspace(-1.0 * halfbox, halfbox, 9)]
    fudgeTicks = fullTicks[1:]

    aspect = "equal"

    # DPI Controlled by user as lower res needed for videos #
    fig, axes = plt.subplots(
        nrows=1, ncols=2, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
    )

    if titleBool is True:
        # Add overall figure plot
        TITLE = (
            r"Redshift $(z) =$"
            + f"{redshift:0.03f} "
            + " "
            + r"$t_{Lookback}=$"
            + f"{tlookback :0.03f} Gyr"
        )
        fig.suptitle(TITLE, fontsize=fontsizeTitle)

    # cmap = plt.get_cmap(CMAP)
    cmap.set_bad(color="grey")

    # -----------#
    # Plot Temperature #
    # -----------#
    # print("pcm1")
    ax1 = axes[0]

    pcm1 = ax1.pcolormesh(
        slice_T["x"],
        slice_T["y"],
        np.transpose(slice_T["grid"]),
        vmin=1e4,
        vmax=10 ** (6.5),
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax1.set_title(f"Temperature Slice", fontsize=fontsize)
    cax1 = inset_axes(ax1, width="5%", height="95%", loc="right")
    fig.colorbar(pcm1, cax=cax1, orientation="vertical").set_label(
        label="T (K)", size=fontsize, weight="bold"
    )
    cax1.yaxis.set_ticks_position("left")
    cax1.yaxis.set_label_position("left")
    cax1.yaxis.label.set_color("white")
    cax1.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax1.set_ylabel(f"{AxesLabels[Axes[1]]}" + " (kpc)", fontsize=fontsize)
    # ax1.set_xlabel(f'{AxesLabels[Axes[0]]}"+" [kpc]"', fontsize = fontsize)
    ax1.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax1)
    plt.xticks(fullTicks)
    plt.yticks(fudgeTicks)

    # -----------#
    # Plot n_H Projection #
    # -----------#
    # print("pcm2")
    ax2 = axes[1]

    # cmapVol = cm.get_cmap("seismic")
    # # bounds = [0.125, 8.0, 64.0]
    # norm = matplotlib.colors.LogNorm(clip=True)
    # pcm2 = ax2.pcolormesh(
    #     slice_vol["x"],
    #     slice_vol["y"],
    #     np.transpose(slice_vol["grid"]),
    #     vmin = 1e-1,
    #     vmax = 1e1,
    #     norm=norm,
    #     cmap=cmapVol,
    #     rasterized=True,
    # )

    cmapVol = cm.get_cmap("seismic")
    bounds = [0.125, 1.0, 8.0, 64.0]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmapVol.N, extend="both")
    pcm2 = ax2.pcolormesh(
        slice_vol["x"],
        slice_vol["y"],
        np.transpose(slice_vol["grid"]),
        norm=norm,
        cmap=cmapVol,
        rasterized=True,
    )

    ax2.set_title(r"Volume Slice", fontsize=fontsize)

    cax2 = inset_axes(ax2, width="5%", height="95%", loc="right")
    fig.colorbar(pcm2, cax=cax2, orientation="vertical").set_label(
        label=r"V (kpc$^{3}$)", size=fontsize, weight="bold"
    )
    cax2.yaxis.set_ticks_position("left")
    cax2.yaxis.set_label_position("left")
    cax2.yaxis.label.set_color("white")
    cax2.tick_params(axis="y", colors="white", labelsize=fontsize)
    # ax2.set_ylabel(f'{AxesLabels[Axes[1]]} "+r" (kpc)"', fontsize=fontsize)
    # ax2.set_xlabel(f'{AxesLabels[Axes[0]]} "+r" (kpc)"', fontsize=fontsize)
    ax2.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax2)
    plt.xticks(fullTicks)
    plt.yticks(fullTicks)


    # print("snapnum")
    # Pad snapnum with zeroes to enable easier video making
    if titleBool is True:
        fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.90)
    else:
        fig.subplots_adjust(wspace=0.0, hspace=0.0, top=0.95)

    # fig.tight_layout()

    SaveSnapNumber = str(snapNumber).zfill(4)
    savePath = savePath + f"Slice_Plot_{int(SaveSnapNumber)}.pdf"

    print(f" Save {savePath}")
    plt.savefig(savePath, transparent=False)
    plt.close()

    print(f" ...done!")

    return

def phases_plot(
    simDict,
    ylabel,
    xlimDict,
    logParameters,
    weightKeys=["mass",
                ],
    fontsize=13,
    fontsizeTitle=14,
    titleBool=True,
    DPI=200,
    xsize=8.0,
    ysize=8.0,
    colourmapMain="plasma",
    Nbins=250,
):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    zlimDict = copy.deepcopy(xlimDict)

    zlimDict.update({"rho_rhomean": {"xmin": 0.25, "xmax": 6.5}})
    zlimDict.update({"T": {"xmin": 3.75, "xmax": 7.0}})
    zlimDict.update({"tcool_tff": {"xmin": -2.5, "xmax": 2.0}})
    zlimDict.update({"gz": {"xmin": -1.0, "xmax": 0.25}})
    zlimDict.update({"Pthermal_Pmagnetic": {"xmin": -2.0, "xmax": 10.0}})

    savePath = f"./Plots/Phases/"
    tmp = "./"

    for savePathChunk in savePath.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass
    # ------------------------------------------------------------------------------#
    #               PLOTTING
    #
    # ------------------------------------------------------------------------------#
    for weightKey in weightKeys:
        print("\n" + f"Starting weightKey {weightKey}")

        zmin = zlimDict[weightKey]["xmin"]
        zmax = zlimDict[weightKey]["xmax"]

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=(xsize, ysize),
            dpi=DPI,
            sharey=True,
            sharex=True,
        )

        currentAx = ax

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        #   Figure 1: Full Cells Data
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        xdataCells = np.log10(
            simDict["rho_rhomean"]
        )
        ydataCells = np.log10(simDict["T"])
        massCells = simDict["mass"]
        try:
            weightDataCells = (
                simDict[weightKey] * massCells
            )
            skipBool = False
        except:
            print(
                f"Variable {weightKey} not found. Skipping plot..."
            )
            skipBool = True
            continue

        if weightKey == "mass":
            finalHistCells, xedgeCells, yedgeCells = np.histogram2d(
                xdataCells, ydataCells, bins=Nbins, weights=massCells
            )
        else:
            mhistCells, _, _ = np.histogram2d(
                xdataCells, ydataCells, bins=Nbins, weights=massCells
            )
            histCells, xedgeCells, yedgeCells = np.histogram2d(
                xdataCells, ydataCells, bins=Nbins, weights=weightDataCells
            )

            finalHistCells = histCells / mhistCells

        finalHistCells[finalHistCells == 0.0] = np.nan
        try:
            if weightKey in logParameters:
                finalHistCells = np.log10(finalHistCells)
        except:
            print(f"Variable {weightKey} not found. Skipping plot...")
            skipBool = True
            continue
        finalHistCells = finalHistCells.T

        xcells, ycells = np.meshgrid(xedgeCells, yedgeCells)

        img1 = currentAx.pcolormesh(
            xcells,
            ycells,
            finalHistCells,
            cmap=colourmapMain,
            vmin=zmin,
            vmax=zmax,
            rasterized=True,
        )
        #
        # img1 = currentAx.imshow(finalHistCells,cmap=colourmapMain,vmin=xmin,vmax=xmax \
        # ,extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')

        currentAx.set_xlabel(
            r"Log10 Density ($ \rho / \langle \rho \rangle $)",
            fontsize=fontsize,
        )
        currentAx.set_ylabel(
            "Log10 Temperatures (K)", fontsize=fontsize)

        currentAx.set_ylim(
            zlimDict["T"]["xmin"], zlimDict["T"]["xmax"])
        currentAx.set_xlim(
            zlimDict["rho_rhomean"]["xmin"], zlimDict["rho_rhomean"]["xmax"])
        currentAx.tick_params(
            axis="both", which="both", labelsize=fontsize)

        currentAx.set_aspect("auto")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        #   Figure: Finishing up
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
        if skipBool == True:
            try:
                tmp = finalHistCells
            except:
                print(
                    f"Variable {weightKey} not found. Skipping plot..."
                )
                continue
            else:
                pass

            #left, bottom, width, height
            # x0,    y0,  delta x, delta y
        cax1 = fig.add_axes([0.925, 0.10, 0.05, 0.80])

        fig.colorbar(img1, cax=cax1, ax=ax, orientation="vertical", pad=0.05).set_label(
            label=ylabel[weightKey], size=fontsize
        )
        cax1.yaxis.set_ticks_position("left")
        cax1.yaxis.set_label_position("left")
        cax1.yaxis.label.set_color("black")
        cax1.tick_params(axis="y", colors="black", labelsize=fontsize)

        if titleBool is True:
            fig.suptitle(
                f"Temperature Density Diagram, weighted by {weightKey}",
                fontsize=fontsizeTitle,
            )

        if titleBool is True:
            plt.subplots_adjust(top=0.875, right=0.8, hspace=0.3, wspace=0.3)
        else:
            plt.subplots_adjust(right=0.8, hspace=0.3, wspace=0.3)

        SaveSnapNumber = str(snapNumber).zfill(4)

        opslaan = (
            savePath
            + f"Phases-Plot-{weightKey}_{int(SaveSnapNumber)}.pdf"
        )
        plt.savefig(opslaan, dpi=DPI, transparent=False)
        print(opslaan)

    return



if __name__ == "__main__":
    # load in the subfind group files
    snap_subfind = load_subfind(snapNumber, dir=loadpath)

    snapGas = gadget_readsnap(
        snapNumber,
        loadpath,
        hdf5=True,
        loadonlytype=[0, 1, 4],
        lazy_load=False,
        # subfind=snap_subfind,
    )

    rotation_matrix = None
    snapGas.calc_sf_indizes(snap_subfind, halolist=[0])
    if rotation_matrix is None:
        rotation_matrix = snapGas.select_halo(snap_subfind, do_rotation=True)
    else:
        snapGas.select_halo(snap_subfind, do_rotation=False)
        snapGas.rotateto(
            rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
        )


    print(
        f"[@{int(snapNumber)}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
    )

    snapGas.calc_sf_indizes(snap_subfind, halolist=[0])
    # snapGas.calc_sf_indizes(snap_subfind, halolist=[0])


    # snapGas.select_halo(snap_subfind, do_rotation=False)
    # --------------------------#
    ##    Units Conversion    ##
    # --------------------------#

    # Convert Units
    ## Make this a seperate function at some point??
    snapGas.pos *= 1e3  # [kpc]
    snapGas.vol *= 1e9  # [kpc^3]
    snapGas.mass *= 1e10  # [Msol]
    snapGas.hrgm *= 1e10  # [Msol]

    snapGas.data["R"] = np.linalg.norm(snapGas.data["pos"], axis=1)

    print(
        f"[@{int(snapNumber)}]: Select stars..."
    )

    whereWind = snapGas.data["age"] < 0.0

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereWind,
        errorString = "Remove Wind from Gas",
        DEBUG = DEBUG,
        )


    Rvir = (snap_subfind.data["frc2"] * 1e3)[int(0)]

    whereOutsideVirial = snapGas.data["R"] > Rvir

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereOutsideVirial,
        errorString = "Remove Outside Virial from Gas",
        DEBUG = DEBUG,
        )

    rmax = 175.0
    boxmax = rmax
    box = [boxmax, boxmax, boxmax]

    # Calculate New Parameters and Load into memory others we want to track
    snapGas = calculate_tracked_parameters(
        snapGas,
        oc.elements,
        oc.elements_Z,
        oc.elements_mass,
        oc.elements_solar,
        oc.Zsolar,
        oc.omegabaryon0,
        snapNumber,
        logParameters = logParameters,
        paramsOfInterest=["T","rho_rhomean"],
        mappingBool=True,
        box=box,
        numthreads=numthreads,
        verbose = False,
    )


    whereDM = snapGas.data["type"] == 1

    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereDM,
        errorString = "Remove DM from Gas",
        DEBUG = DEBUG,
        )

    whereStars = snapGas.data["type"] == 4
    snapGas = remove_selection(
        snapGas,
        removalConditionMask = whereStars,
        errorString = "Remove Stars from Gas",
        DEBUG = DEBUG
        )

    # Redshift
    redshift = snapGas.redshift  # z
    aConst = 1.0 / (1.0 + redshift)  # [/]

    # Get lookback time in Gyrs
    # [0] to remove from numpy array for purposes of plot title
    lookback = snapGas.cosmology_get_lookback_time_from_a(np.array([aConst]))[
        0
    ]  # [Gyrs]

    snapGas.data["Redshift"] = np.array([redshift])
    snapGas.data["Lookback"] = np.array([lookback])
    snapGas.data["Snap"] = np.array([snapNumber])
    snapGas.data["Rvir"] = np.array([Rvir])

    print(
        f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
    )
    # Make normal dictionary form of snapGas
    out = {}
    for key, value in snapGas.data.items():
        if value is not None:
            out.update({key: copy.deepcopy(value)})

    print(
        f"[@{int(snapNumber)}]: Finishing process..."
    )

    print(
        f"[@{int(snapNumber)}]: Slice plot"
    )

    plot_slices(snapGas,
        snapNumber,
        boxsize=Rvir*0.95
    )

    print(
        f"[@{int(snapNumber)}]: Phases plot"
    )

    phases_plot(
        out,
        ylabel,
        xlimDict,
        logParameters
    )

    print(
        f"[@{int(snapNumber)}]: Done"
    )
