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

ageWindow = 1.5 #(Gyr) before current snapshot SFR evaluation
windowBins = 0.100 #(Gyr) size of ageWindow Bins. Ignored if ageWindow is None
Nbins = 250
snapStart = 100
snapEnd = 102
DEBUG = False

loadPathBase = "/home/cosmos/c1838736/Auriga/level5_cgm/"
loadDirectories = [
    "h5_2kpc",
    "snapshot-restart-of-2kpc/h5_1kpc_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy_snapshot-restart-of-2kpc",
    "snapshot-restart-of-2kpc/h5_hy-v2_snapshot-restart-of-2kpc"
]

simulations = []
savePaths = []

for dir in loadDirectories:
    loadpath = loadPathBase+dir+"/output/"
    simulations.append(loadpath)
    savepath = "./" + dir + "/"
    savePaths.append(savepath)


numthreads = 18

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


def plot_slices(snap,
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
    savePathBase = "./",
):
    savePath = savePathBase + f"Plots/Slices/"
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
    # print(np.unique(snap.type))
    print("\n" + f"Projection 1 of {nprojections}")

    slice_T = snap.get_Aslice(
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

    slice_vol = snap.get_Aslice(
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
    ysize = 5.0
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    if titleBool is True:
        # Redshift
        redshift = snap.redshift  # z
        aConst = 1.0 / (1.0 + redshift)  # [/]

        # [0] to remove from numpy array for purposes of plot title
        tlookback = snap.cosmology_get_lookback_time_from_a(np.array([aConst]))[
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
    # ax1.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax1)
    plt.xticks(fullTicks)
    plt.yticks(fudgeTicks)

    # -----------#
    # Plot n_H Projection #
    # -----------#
    # print("pcm2")
    ax2 = axes[1]

    cmapVol = cm.get_cmap("seismic")
    norm = matplotlib.colors.LogNorm(clip=True)
    pcm2 = ax2.pcolormesh(
        slice_vol["x"],
        slice_vol["y"],
        np.transpose(slice_vol["grid"]),
        vmin = 5e-3,#5e-1,
        vmax = 5e3,#2e1,
        norm=norm,
        cmap=cmapVol,
        rasterized=True,
    )

    # cmapVol = cm.get_cmap("seismic")
    # bounds = [0.5, 2.0, 4.0, 16.0]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmapVol.N, extend="both")
    # pcm2 = ax2.pcolormesh(
    #     slice_vol["x"],
    #     slice_vol["y"],
    #     np.transpose(slice_vol["grid"]),
    #     norm=norm,
    #     cmap=cmapVol,
    #     rasterized=True,
    # )

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
    # ax2.set_aspect(aspect)

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
    savePath = savePath + f"Slice_Plot_{int(SaveSnapNumber)}.pdf" #_binary-split

    print(f" Save {savePath}")
    plt.savefig(savePath, transparent=False)
    plt.close()

    print(f" ...done!")

    return

def plot_slices_quad(snap,
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
    savePathBase = "./",
):
    savePath = savePathBase + "Plots/Slices/"

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
    # PLOTTING TIME
    # Set plot figure sizes
    xsize = 10.0
    ysize = 10.0
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    if titleBool is True:
        # Redshift
        redshift = snap.redshift  # z
        aConst = 1.0 / (1.0 + redshift)  # [/]

        # [0] to remove from numpy array for purposes of plot title
        tlookback = snap.cosmology_get_lookback_time_from_a(np.array([aConst]))[
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

    nprojections = 4
    # print(np.unique(snap.type))
    print("\n" + f"[@{int(snapNumber)}]: Projection 1 of {nprojections}")
    slice_T = snap.get_Aslice(
        "T",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 2 of {nprojections}")

    slice_tcool = snap.get_Aslice(
        "tcool",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 3 of {nprojections}")

    slice_nH = snap.get_Aslice(
        "n_H",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )

    print("\n" + f"[@{int(snapNumber)}]: Projection 4 of {nprojections}")

    # slice_gz = snap.get_Aslice(
    #     "gz",
    #     box=[boxsize, boxsize],
    #     center=imgcent,
    #     nx=int(boxsize / pixres),
    #     ny=int(boxsize / pixres),
    #     axes=Axes,
    #     proj=False,
    #     numthreads=numthreads,
    # )

    slice_cl = snap.get_Aslice(
        "cool_length",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        axes=Axes,
        proj=False,
        numthreads=numthreads,
    )


    fig, axes = plt.subplots(
        nrows=2, ncols=2, figsize=(xsize, ysize), dpi=DPI, sharex=True, sharey=True
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
    ax1 = axes[0,0]

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
    # ax1.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax1)
    plt.xticks(fullTicks)
    plt.yticks(fudgeTicks)

    # -----------#
    # Plot n_H Projection #
    # -----------#
    # print("pcm2")
    ax2 = axes[0,1]

    pcm2 = ax2.pcolormesh(
        slice_tcool["x"],
        slice_tcool["y"],
        np.transpose(slice_tcool["grid"]),
        vmin = (10)**(-3.5),
        vmax = 1e2,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    # cmapVol = cm.get_cmap("seismic")
    # bounds = [0.5, 2.0, 4.0, 16.0]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmapVol.N, extend="both")
    # pcm2 = ax2.pcolormesh(
    #     slice_vol["x"],
    #     slice_vol["y"],
    #     np.transpose(slice_vol["grid"]),
    #     norm=norm,
    #     cmap=cmapVol,
    #     rasterized=True,
    # )

    ax2.set_title(r"Cooling Time Slice", fontsize=fontsize)

    cax2 = inset_axes(ax2, width="5%", height="95%", loc="right")
    fig.colorbar(pcm2, cax=cax2, orientation="vertical").set_label(
        label=r"t$_{cool}$ (Gyr)", size=fontsize, weight="bold"
    )
    cax2.yaxis.set_ticks_position("left")
    cax2.yaxis.set_label_position("left")
    cax2.yaxis.label.set_color("white")
    cax2.tick_params(axis="y", colors="white", labelsize=fontsize)
    # ax2.set_ylabel(f'{AxesLabels[Axes[1]]} "+r" (kpc)"', fontsize=fontsize)
    # ax2.set_xlabel(f'{AxesLabels[Axes[0]]} "+r" (kpc)"', fontsize=fontsize)
    # ax2.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax2)
    plt.xticks(fullTicks)
    plt.yticks(fullTicks)

    # -----------#
    # Plot Metallicity #
    # -----------#
    # print("pcm3")
    ax3 = axes[1, 0]

    pcm3 = ax3.pcolormesh(
        slice_cl["x"],
        slice_cl["y"],
        np.transpose(slice_cl["grid"]),
        # vmin=10**(-0.5),
        # vmax=10**(1.0),
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    # ax3.set_title(f"Metallicity Slice", y=-0.2, fontsize=fontsize)
    #
    # cax3 = inset_axes(ax3, width="5%", height="95%", loc="right")
    # fig.colorbar(pcm3, cax=cax3, orientation="vertical").set_label(
    #     label=r"$Z/Z_{\odot}$", size=fontsize, weight="bold"
    # )
    ax3.set_title(f"Cooling Length Slice", y=-0.2, fontsize=fontsize)

    cax3 = inset_axes(ax3, width="5%", height="95%", loc="right")
    fig.colorbar(pcm3, cax=cax3, orientation="vertical").set_label(
        label=r"$l_{cool}$ (kpc)", size=fontsize, weight="bold"
    )

    cax3.yaxis.set_ticks_position("left")
    cax3.yaxis.set_label_position("left")
    cax3.yaxis.label.set_color("white")
    cax3.tick_params(axis="y", colors="white", labelsize=fontsize)

    ax3.set_ylabel(f"{AxesLabels[Axes[1]]} " + r" (kpc)", fontsize=fontsize)
    ax3.set_xlabel(f"{AxesLabels[Axes[0]]} " + r" (kpc)", fontsize=fontsize)

    # ax3.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax3)
    plt.xticks(fullTicks)
    plt.yticks(fullTicks)

    # -----------#
    # Plot Magnetic Field Projection #
    # -----------#
    # print("pcm4")
    ax4 = axes[1, 1]

    pcm4 = ax4.pcolormesh(
        slice_nH["x"],
        slice_nH["y"],
        np.transpose(slice_nH["grid"]),
        vmin=1e-7,
        vmax=1e-1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax4.set_title(r"HI Number Density Slice",
                  y=-0.2, fontsize=fontsize)

    cax4 = inset_axes(ax4, width="5%", height="95%", loc="right")
    fig.colorbar(pcm4, cax=cax4, orientation="vertical").set_label(
        label=r"n$_{H}$ (cm$^{-3}$)", size=fontsize, weight="bold"
    )
    cax4.yaxis.set_ticks_position("left")
    cax4.yaxis.set_label_position("left")
    cax4.yaxis.label.set_color("white")
    cax4.tick_params(axis="y", colors="white", labelsize=fontsize)

    # ax4.set_ylabel(f'{AxesLabels[Axes[1]]} "+r" (kpc)"', fontsize=fontsize)
    ax4.set_xlabel(f"{AxesLabels[Axes[0]]} " + r" (kpc)", fontsize=fontsize)
    # ax4.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax4)
    plt.xticks(fudgeTicks)
    plt.yticks(fullTicks)

    # print("snapnum")
    # Pad snapnum with zeroes to enable easier video making
    if titleBool is True:
        fig.subplots_adjust(wspace=0.0 ,hspace=0.0, top=0.90)
    else:
        fig.subplots_adjust(wspace=0.0 ,hspace=0.0, top=0.95)

    # fig.tight_layout()

    SaveSnapNumber = str(snapNumber).zfill(4)
    savePath = savePath + f"Slice_Plot_Quad_{int(SaveSnapNumber)}.pdf" #_binary-split

    print(f" Save {savePath}")
    plt.savefig(savePath, transparent=False)
    plt.close()

    print(f" ...done!")

    return

def plot_projections(snap,
    snapNumber,
    fontsize=13,
    fontsizeTitle=14,
    titleBool=True,
    Axes=[0, 1],
    zAxis = [2],
    boxsize=400.0,
    boxlos=50.0,
    pixreslos=0.3,
    pixres=0.3,
    DPI=200,
    CMAP=None,
    numthreads=10,
    savePathBase = "./",
):

    savePath = savePathBase + "Plots/Projections/"
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

    nprojections = 3
    # print(np.unique(snap.type))
    print("\n" + f"Projection 1 of {nprojections}")

    proj_T = snap.get_Aslice(
        "Tdens",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numthreads,
    )

    print("\n" + f"Projection 2 of {nprojections}")

    proj_dens = snap.get_Aslice(
        "rho_rhomean",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numthreads,
    )

    print("\n" + f" Projection 3 of {nprojections}")

    proj_vol = snap.get_Aslice(
        "vol",
        box=[boxsize, boxsize],
        center=imgcent,
        nx=int(boxsize / pixres),
        ny=int(boxsize / pixres),
        nz=int(boxlos / pixreslos),
        boxz=boxlos,
        axes=Axes,
        proj=True,
        numthreads=numthreads,
    )

    # ------------------------------------------------------------------------------#
    # PLOTTING TIME
    # Set plot figure sizes
    xsize = 10.0
    ysize = 5.0
    # Define halfsize for histogram ranges which are +/-
    halfbox = boxsize / 2.0

    if titleBool is True:
        # Redshift
        redshift = snap.redshift  # z
        aConst = 1.0 / (1.0 + redshift)  # [/]

        # [0] to remove from numpy array for purposes of plot title
        tlookback = snap.cosmology_get_lookback_time_from_a(np.array([aConst]))[
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
        proj_T["x"],
        proj_T["y"],
        np.transpose(proj_T["grid"] / proj_dens["grid"]),
        vmin=1e4,
        vmax=10 ** (6.5),
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax1.set_title(f"Temperature Projection", fontsize=fontsize)
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
    # ax1.set_aspect(aspect)

    # Fudge the tick labels...
    plt.sca(ax1)
    plt.xticks(fullTicks)
    plt.yticks(fudgeTicks)

    # -----------#
    # Plot n_H Projection #
    # -----------#
    # print("pcm2")
    ax2 = axes[1]

    cmapVol = cm.get_cmap("seismic")
    norm = matplotlib.colors.LogNorm(clip=True)
    pcm2 = ax2.pcolormesh(
        proj_vol["x"],
        proj_vol["y"],
        np.transpose(proj_vol["grid"]) / int(boxlos / pixreslos),
        vmin = 5e-3,#5e-1,
        vmax = 5e3,#2e1,
        norm=norm,
        cmap=cmapVol,
        rasterized=True,
    )

    # cmapVol = cm.get_cmap("seismic")
    # bounds = [0.5, 2.0, 4.0, 16.0]
    # norm = matplotlib.colors.BoundaryNorm(bounds, cmapVol.N, extend="both")
    # pcm2 = ax2.pcolormesh(
    #     proj_vol["x"],
    #     proj_vol["y"],
    #     np.transpose(proj_vol["grid"]),
    #     norm=norm,
    #     cmap=cmapVol,
    #     rasterized=True,
    # )

    ax2.set_title(r"Volume Projection", fontsize=fontsize)

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
    # ax2.set_aspect(aspect)

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
    savePath = savePath + f"Projection_Plot_{int(SaveSnapNumber)}.pdf" #_binary-split

    print(f" Save {savePath}")
    plt.savefig(savePath, transparent=False)
    plt.close()

    print(f" ...done!")

    return


def hist_plot_xyz(
    simDict,
    ylabel,
    xlimDict,
    logParameters,
    yParams = ["T"],
    xParams = ["rho_rhomean","R"],
    weightKeys = ["mass","vol"],
    axisLimsBool = True,
    fontsize=13,
    fontsizeTitle=14,
    titleBool=True,
    DPI=200,
    xsize=8.0,
    ysize=8.0,
    colourmapMain="plasma",
    Nbins=250,
    savePathBase = "./",
):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    zlimDict = copy.deepcopy(xlimDict)


    savePath = savePathBase + "Plots/Phases/"
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
    for yParam in yParams:
        print("\n"+"-----")
        print(f"Starting yParam {yParam}")
        for xParam in xParams:
            print("\n"+f"Starting xParam {xParam}")
            for weightKey in weightKeys:
                print("\n"+f"Starting weightKey {weightKey}")

                if weightKey == xParam:
                    print("\n" + f"WeightKey same as xParam! Skipping...")
                    skipBool = True
                    continue

                if weightKey == yParam:
                    print("\n" + f"WeightKey same as yParam! Skipping...")
                    skipBool = True
                    continue

                if xParam == yParam:
                    print("\n" + f"yParam same as xParam! Skipping...")
                    skipBool = True
                    continue

                if np.all(np.isin(np.array(["tcool","theat"]),np.array([xParam,yParam,weightKey]))) == True:
                    print("\n" + f"tcool and theat aren't compatible! Skipping...")
                    skipBool = True
                    continue

                if axisLimsBool:
                    try:
                        zmin = zlimDict[weightKey]["xmin"]
                        zmax = zlimDict[weightKey]["xmax"]
                        zlimBool = True
                    except:
                        zlimBool = False

                    try:
                        tmp = zlimDict[xParam]["xmin"]
                        tmp = zlimDict[xParam]["xmax"]
                        xlimBool = True
                    except:
                        xlimBool = False

                    try:
                        tmp = zlimDict[yParam]["xmin"]
                        tmp = zlimDict[yParam]["xmax"]
                        ylimBool = True
                    except:
                        ylimBool = False
                else:
                    zlimBool = False
                    xlimBool = False
                    ylimBool = False


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
                try:
                    if xParam in logParameters:
                        xx = np.log10(simDict[xParam])
                    else:
                        xx = simDict[xParam]
                except:
                    print("\n"+f"xParam of {xParam} data not found! Skipping...")
                    skipBool = True
                    continue

                try:
                    if yParam in logParameters:
                        yy = np.log10(simDict[yParam])
                    else:
                        yy = simDict[yParam]
                except:
                    print("\n"+f"yParam of {yParam} data not found! Skipping...")
                    skipBool = True
                    continue

                try:
                    xmin, xmax =(
                        zlimDict[xParam]["xmin"], zlimDict[xParam]["xmax"]
                    )
                except:
                    xmin, xmax, = ( np.nanmin(xx), np.nanmax(xx))

                try:
                    ymin, ymax =(
                        zlimDict[yParam]["xmin"], zlimDict[yParam]["xmax"]
                    )
                except:
                    ymin, ymax, = ( np.nanmin(yy), np.nanmax(yy))

                xdataCells = xx[np.where((xx>=xmin)&(xx<=xmax)&(yy>=ymin)&(yy<=ymax)&(np.isfinite(xx)==True)&(np.isfinite(yy)==True)) [0]]
                ydataCells = yy[np.where((xx>=xmin)&(xx<=xmax)&(yy>=ymin)&(yy<=ymax)&(np.isfinite(xx)==True)&(np.isfinite(yy)==True))[0]]

                massCells = ( simDict["mass"][
                    np.where((xx>=xmin)&(xx<=xmax)
                    &(yy>=ymin)&(yy<=ymax)&
                    (np.isfinite(xx)==True)&(np.isfinite(yy)==True))
                    [0]]
                )
                try:
                    weightDataCells = (
                        simDict[weightKey][
                        np.where((xx>=xmin)&(xx<=xmax)
                        &(yy>=ymin)&(yy<=ymax)&(np.isfinite(xx)==True)&(np.isfinite(yy)==True))
                        [0]] * massCells
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

                if zlimBool is True:
                    img1 = currentAx.pcolormesh(
                        xcells,
                        ycells,
                        finalHistCells,
                        cmap=colourmapMain,
                        vmin=zmin,
                        vmax=zmax,
                        rasterized=True,
                    )
                else:
                    img1 = currentAx.pcolormesh(
                        xcells,
                        ycells,
                        finalHistCells,
                        cmap=colourmapMain,
                        rasterized=True,
                    )
                #
                # img1 = currentAx.imshow(finalHistCells,cmap=colourmapMain,vmin=xmin,vmax=xmax \
                # ,extent=[np.min(xedgeCells),np.max(xedgeCells),np.min(yedgeCells),np.max(yedgeCells)],origin='lower')

                currentAx.set_xlabel(
                    ylabel[xParam],
                    fontsize=fontsize,
                )
                currentAx.set_ylabel(
                    ylabel[yParam],
                    fontsize=fontsize
                )

                if ylimBool is True:
                    currentAx.set_ylim(
                        zlimDict[yParam]["xmin"], zlimDict[yParam]["xmax"])
                else:
                    currentAx.set_ylim(np.nanmin(yedgeCells),np.nanmax(yedgeCells))

                if xlimBool is True:
                    if xParam == "vol":
                        currentAx.set_xlim(zlimDict[xParam]["xmax"],zlimDict[xParam]["xmin"])
                    else:
                        currentAx.set_xlim(zlimDict[xParam]["xmin"],zlimDict[xParam]["xmax"])
                else:
                    if xParam == "vol":
                        currentAx.set_xlim(np.nanmax(xedgeCells),np.nanmin(xedgeCells))
                    else:
                        currentAx.set_xlim(np.nanmin(xedgeCells),np.nanmax(xedgeCells))
                    # zlimDict["rho_rhomean"]["xmin"], zlimDict["rho_rhomean"]["xmax"])
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
                        f"{yParam} vs. {xParam} Diagram, weighted by {weightKey}",
                        fontsize=fontsizeTitle,
                    )

                if titleBool is True:
                    plt.subplots_adjust(top=0.875, right=0.8, hspace=0.3, wspace=0.3)
                else:
                    plt.subplots_adjust(right=0.8, hspace=0.3, wspace=0.3)

                SaveSnapNumber = str(snapNumber).zfill(4)

                opslaan = (
                    savePath
                    + f"Phases-Plot__{yParam}-vs-{xParam}_weighted-by-{weightKey}_{int(SaveSnapNumber)}.pdf"
                )
                plt.savefig(opslaan, dpi=DPI, transparent=False)
                print(opslaan)

    return

def pdf_versus_plot(
    dataDict,
    ylabel,
    xlimDict,
    logParameters,
    snapNumber,
    weightKeys = ['mass'],
    xParams = ["T"],
    axisLimsBool = True,
    titleBool=False,
    densityBool=True,
    DPI=150,
    xsize=6.0,
    ysize=6.0,
    fontsize=13,
    fontsizeTitle=14,
    Nbins=250,
    ageWindow=None,
    cumulative = False,
    savePathBase = "./",
    saveCurve = False,
    SFR = False,
    byType = False,
):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    limDict = copy.deepcopy(xlimDict)


    savePath = savePathBase + "Plots/PDFs/"
    tmp = "./"

    for savePathChunk in savePath.split("/")[1:-1]:
        tmp += savePathChunk + "/"
        try:
            os.mkdir(tmp)
        except:
            pass
        else:
            pass

    if SFR is True:
        weightKeys = ["gima"]
        xParams = ["age"]

    if byType is True:
        uniqueTypes = np.unique(dataDict["type"])
        for tp in uniqueTypes:
            print("Starting type ",tp)
            whereNotType = dataDict["type"] != tp

            tpData = remove_selection(
                copy.deepcopy(dataDict),
                removalConditionMask = whereNotType,
                errorString = "byType PDF whereNotType",
                DEBUG = False,
                )

            pdf_versus_plot(
                dataDict = tpData,
                ylabel = ylabel,
                xlimDict = xlimDict,
                logParameters = logParameters,
                snapNumber = snapNumber,
                weightKeys = weightKeys,
                xParams = xParams,
                axisLimsBool = axisLimsBool,
                titleBool = titleBool,
                densityBool = densityBool,
                DPI = DPI,
                xsize = xsize,
                ysize = ysize,
                fontsize = fontsize,
                fontsizeTitle = fontsizeTitle,
                Nbins = Nbins,
                ageWindow = ageWindow,
                cumulative = cumulative,
                savePathBase = savePathBase+f"type{int(tp)}/",
                saveCurve = saveCurve,
                SFR = SFR,
                byType = False,
            )
        return

    for weightKey in weightKeys:
        print("-----")
        print("")
        print(f"Starting {weightKey} weighted!")
        for analysisParam in xParams:
            print("")
            print(f"Starting {analysisParam} plots!")
            fig, ax = plt.subplots(
                nrows=1,
                ncols=1,
                sharex=True,
                sharey=True,
                figsize=(xsize, ysize),
                dpi=DPI,
            )
            # Create a plot for each Temperature
            skipBool = False
            try:
                plotData = dataDict[analysisParam].copy()
                weightsData = dataDict[weightKey].copy()
                skipBool = False
            except:
                print(
                    f"Variable {analysisParam} not found. Skipping plot..."
                )
                skipBool = True
                continue

            colour = "blue"

            if analysisParam in logParameters:
                tmpPlot = np.log10(plotData).copy()
            else:
                tmpPlot = plotData.copy()

            tmpWeights = weightsData.copy()

            SFRBool = False
            if (weightKey == "gima")&(analysisParam=="age"):
                SFRBool = True

            whereAgeBelowLimit = np.full(shape=np.shape(tmpPlot),fill_value=True)
            if ageWindow is not None:
                if SFRBool is True:
                    print("Minimum age detected = ", np.nanmin(tmpPlot), "Gyr")
                    # minAge = np.nanmin(tmpPlot) + ((np.nanmax(tmpPlot) - np.nanmin(tmpPlot))*ageWindow)
                    maxAge = np.nanmin(tmpPlot)+ageWindow
                    print("Maximum age for plotting = ", maxAge, "Gyr")

                    whereAgeBelowLimit = tmpPlot<=maxAge
                    print("Number of data points meeting age = ",np.shape(np.where(whereAgeBelowLimit==True)[0])[0])
                else:
                    print("[@pdf_versus_plot]: ageWindow not None, but SFR plot not detected. ageWindow will be ignored...")

            if axisLimsBool is True:
                try:
                    xmin, xmax =(
                        limDict[analysisParam]["xmin"],
                        limDict[analysisParam]["xmax"]
                    )
                except:
                    xmin, xmax, = ( np.nanmin(tmpPlot[np.where(whereAgeBelowLimit==True)[0]]),
                     np.nanmax(tmpPlot[np.where(whereAgeBelowLimit==True)[0]]))

                try:
                    whereData = np.where((np.isfinite(tmpPlot)==True)
                    & (np.isfinite(tmpWeights)==True)
                    & (tmpPlot>=xmin)
                    & (tmpPlot<=xmax)
                    & (whereAgeBelowLimit == True)
                    )[0]
                except:
                    whereData = np.where((np.isfinite(tmpPlot)==True)
                    & (np.isfinite(tmpWeights)==True)
                    & (whereAgeBelowLimit == True)
                    )[0]
            else:
                whereData = np.where((np.isfinite(tmpPlot)==True)
                & (np.isfinite(tmpWeights)==True)
                & (whereAgeBelowLimit == True)
                )[0]


            plotData = tmpPlot[whereData]
            weightsData = tmpWeights[whereData]

            try:
                xmin = np.nanmin(plotData)
                xmax = np.nanmax(plotData)
                skipBool = False
            except:
                print(
                    f"Variable {analysisParam} not found. Skipping plot...")
                skipBool = True
                continue

            if (
                (np.isfinite(xmin) == False)
                or (np.isfinite(xmax) == False)
                or (np.isfinite(np.nanmin(weightsData)) == False)
                or (np.isfinite(np.nanmin(weightsData)) == False)
            ):
                # print()
                print("Data All Inf/NaN! Skipping entry!")
                skipBool = True
                continue

            xBins = np.linspace(
                start=xmin, stop=xmax, num=Nbins)

            currentAx = ax


            hist, bin_edges = np.histogram(
                plotData,
                bins=xBins,
                weights=weightsData,
            )

            if (SFRBool is True):#|(analysisParam == "R"):
                hist = np.flip(hist)

            if cumulative is True:
                hist = np.cumsum(hist)

            if SFRBool is True:
                delta = np.mean(np.diff(xBins))
                if cumulative is False:
                    hist = hist/(delta*1e9) # convert to SFR per yr

            if (SFRBool is True):#|(analysisParam == "R"):
                xBins = np.flip(xBins)
                bin_edges = np.flip(bin_edges)

            if weightKey in logParameters:
                if weightKey != "mass":
                    hist[hist == 0.0] = np.nan
                    hist = np.log10(hist)

            weightsSumTotal = np.cumsum(weightsData)[-1]


            if np.all(np.isfinite(hist)==False) == True:
                print("Hist All Inf/NaN! Skipping entry!")
                continue

            try:
                ymin = np.nanmin(hist[np.isfinite(hist)])
                ymax = np.nanmax(hist[np.isfinite(hist)])
                skipBool = False
            except:
                print(
                    f"Variable {analysisParam} not found. Skipping plot...")
                skipBool = True
                continue
            xFromBins = np.array(
                [
                    (x1 + x2) / 2.0
                    for (x1, x2) in zip(bin_edges[:-1], bin_edges[1:])
                ]
            )

            currentAx.plot(
                xFromBins,
                hist,
                color=colour,
                linestyle="solid",
                label = f"Sum total of {weightKey} = {weightsSumTotal:.2e}"
            )

            currentAx.xaxis.set_minor_locator(AutoMinorLocator())
            currentAx.yaxis.set_minor_locator(AutoMinorLocator())
            currentAx.tick_params(
                axis="both", which="both", labelsize=fontsize
            )

            ylabel_prefix = ""
            if cumulative is True:
                ylabel_prefix = "Cumulative "
            if weightKey == "mass":
                currentAx.set_ylabel(ylabel_prefix+r"Mass (M$_{\odot}$)", fontsize=fontsize)
            else:
                currentAx.set_ylabel(
                ylabel_prefix+ylabel[weightKey], fontsize=fontsize)


            if titleBool is True:
                fig.suptitle(
                    ylabel_prefix + f"PDF of"
                    + "\n"
                    + f" {weightKey} vs {analysisParam}",
                    fontsize=fontsizeTitle,
                )

            # Only give 1 x-axis a label, as they sharex

            ax.set_xlabel(ylabel[analysisParam], fontsize=fontsize)

            if (skipBool == True):
                print(
                    f"Variable {analysisParam} not found. Skipping plot...")
                continue

            if axisLimsBool is True:
                try:
                    finalxmin = max(
                        np.nanmin(xmin), xlimDict[analysisParam]["xmin"]
                    )
                    finalxmax = min(
                        np.nanmax(xmax), xlimDict[analysisParam]["xmax"]
                    )
                except:
                    finalxmin = xmin
                    finalxmax = xmax
            else:
                finalxmin = xmin
                finalxmax = xmax

            if (
                (np.isinf(finalxmax) == True)
                or (np.isinf(finalxmin) == True)
                or (np.isnan(finalxmax) == True)
                or (np.isnan(finalxmin) == True)
            ):
                print("Data All Inf/NaN! Skipping entry!")
                continue

            if weightKey == "mass":
                if (SFRBool is False):
                    if (cumulative is True):
                        try:
                            finalymin = xlimDict["mass-pdf"]["xmin"]
                            finalymax = xlimDict["mass-pdf"]["xmax"]
                        except:
                            try:
                                finalymin = 0.0
                                finalymax = np.nanmax(ymax)
                            except:
                                print("Data All Inf/NaN! Skipping entry!")
                                continue
                    else:
                        try:
                            finalymin = 0.0
                            finalymax = np.nanmax(ymax)
                        except:
                            print("Data All Inf/NaN! Skipping entry!")
                            continue
                else:
                    try:
                        finalymin = 0.0
                        finalymax = np.nanmax(ymax)
                    except:
                        print("Data All Inf/NaN! Skipping entry!")
                        continue
            else:
                try:
                    finalymin = np.nanmin(ymin)
                    finalymax = np.nanmax(ymax)
                except:
                    print("Data All Inf/NaN! Skipping entry!")
                    continue

            custom_xlim = (finalxmin, finalxmax)
            custom_ylim = (finalymin, finalymax)
            plt.setp(ax, xlim=custom_xlim, ylim=custom_ylim)
            ax.legend(loc="upper left", fontsize=fontsize)

            # plt.tight_layout()
            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1, left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1, left=0.15)

            if cumulative is True:
                tmp2 = savePath +"Cumulative-"
            else:
                tmp2 = savePath
            if SFRBool is True:
                opslaan = tmp2 + f"SFR_{snapNumber}"

            else:
                opslaan = tmp2 + f"{weightKey}-{analysisParam}-PDF_{snapNumber}"

            plt.savefig(opslaan + ".pdf", dpi=DPI, transparent=False)
            print(opslaan)
            plt.close()

            if saveCurve is True:
                out = {"data":{"x" : xFromBins, "y" : hist}}
                hdf5_save(opslaan+"_data.h5",out)
    return


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
                loadonlytype=[0, 1, 2, 3, 4, 5],
                lazy_load=False,
                subfind=snap_subfind,
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
                paramsOfInterest=["R","T","Tdens","rho_rhomean","n_H","gz","tcool","cool_length"],
                mappingBool=True,
                box=box,
                numthreads=numthreads,
                verbose = False,
            )

            print(
                f"[@{int(snapNumber)}]: Remove beyond 1.5 x Virial Radius..."
            )

            whereOutsideVirial = snap.data["R"] > Rvir*1.50#*1.5

            xlimDict["R"]["xmax"] = Rvir*1.50

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

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snaptmp.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})


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

            )

            print(
                f"[@{int(snapNumber)}]: By Type PDF of mass vs R plot..."
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
                byType = True,
            )

            print(
                f"[@{int(snapNumber)}]: By Type Cumulative PDF of mass vs R plot..."
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
                byType = True,
            )

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
                f"[@{int(snapNumber)}]: PDF of gas (mass vs T or vol) plot"
            )

            pdf_versus_plot(
                out,
                ylabel,
                xlimDict,
                logParameters,
                snapNumber,
                weightKeys = ['mass'],
                xParams = ["T","vol"],
                savePathBase = savePathBase,
                saveCurve = True,
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
                xParams = ["T","vol"],
                savePathBase = savePathBase,
                cumulative = True,
                saveCurve = True,
            )


            print(
                f"[@{int(snapNumber)}]: Slice plot"
            )

            plot_slices(snap,
                snapNumber,
                pixres=0.1*1.5,
                boxsize=Rvir*1.50*2.0,
                numthreads=numthreads,
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Slice plot Quad"
            )

            plot_slices_quad(snap,
                snapNumber,
                pixres=0.1*1.5,
                boxsize=Rvir*1.50*2.0,
                numthreads=numthreads,
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Projection plot"
            )

            plot_projections(snap,
                snapNumber,
                boxlos=50.0,
                pixreslos=0.3*1.5,
                pixres=0.3*1.5,
                boxsize=Rvir*1.50*2.0,
                numthreads=numthreads,
                savePathBase = savePathBase,
            )

            # print(
            #     f"[@{int(snapNumber)}]: Remove beyond 1.2 x Virial Radius..."
            # )
            #
            # whereOutsideVirial = snap.data["R"] > Rvir*1.20#*2.0
            #
            # xlimDict["R"]["xmax"] = Rvir*1.20#*1.5
            #
            # snaptmp = remove_selection(
            #     snap,
            #     removalConditionMask = whereOutsideVirial,
            #     errorString = "Remove Outside Virial from Gas",
            #     DEBUG = DEBUG,
            #     )

            print(
                f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
            )
            # Make normal dictionary form of snap
            out = {}
            for key, value in snaptmp.data.items():
                if value is not None:
                    out.update({key: copy.deepcopy(value)})


            print(
                f"[@{int(snapNumber)}]: Hist_plot_xyz plot"
            )

            hist_plot_xyz(
                out,
                ylabel,
                xlimDict,
                logParameters,
                yParams = ["T"],
                xParams = ["R","rho_rhomean","vol", "n_H"],
                weightKeys = ["mass"],
                savePathBase = savePathBase,
            )

            print(
                f"[@{int(snapNumber)}]: Done"
            )
            plt.close("all")
        print("finished sim:", loadpath)
    print("Finished fully! :)")
