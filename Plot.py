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

snapStart = 100
snapEnd = 109
loadpath = "/home/universe/c1838736/Auriga/level5_cgm/h5_hybrid-dev/output/"
numthreads = 8

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
}

xlimDict = {
    "R": {"xmin": 0.0, "xmax": 175.0},
    "mass": {"xmin": 4.0, "xmax": 9.0},
    "L": {"xmin": 3.0, "xmax": 4.5},
    "T": {"xmin": 3.75, "xmax": 7.0},
    "n_H": {"xmin": -5.5, "xmax": -0.5},
    "B": {"xmin": -2.5, "xmax": 1.0},
    "vrad": {"xmin": -150.0, "xmax": 150.0},
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
}

logParameters = ["dens","ndens","rho_rhomean","csound","T","n_H","B","gz","L","P_thermal","P_magnetic","P_kinetic","P_tot","Pthermal_Pmagnetic", "P_CR", "PCR_Pthermal","gah","Grad_T","Grad_n_H","Grad_bfld","Grad_P_CR","tcool","theat","tcross","tff","tcool_tff","mass","gima","vol"]


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
    ysize = 5.0
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

def plot_slices_quad(snapGas,
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

    nprojections = 4
    # print(np.unique(snapGas.type))
    print("\n" + f"[@{int(snapNumber)}]: Projection 1 of {nprojections}")
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

    print("\n" + f"[@{int(snapNumber)}]: Projection 2 of {nprojections}")

    slice_tcool = snapGas.get_Aslice(
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

    slice_nH = snapGas.get_Aslice(
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

    slice_gz = snapGas.get_Aslice(
        "gz",
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
        slice_gz["x"],
        slice_gz["y"],
        np.transpose(slice_gz["grid"]),
        vmin=1e-2,
        vmax=1e1,
        norm=matplotlib.colors.LogNorm(),
        cmap=cmap,
        rasterized=True,
    )

    ax3.set_title(f"Metallicity Slice", y=-0.2, fontsize=fontsize)

    cax3 = inset_axes(ax3, width="5%", height="95%", loc="right")
    fig.colorbar(pcm3, cax=cax3, orientation="vertical").set_label(
        label=r"$Z/Z_{\odot}$", size=fontsize, weight="bold"
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

def plot_projections(snapGas,
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
):
    savePath = f"./Plots/Projections/"
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
    # print(np.unique(snapGas.type))
    print("\n" + f"Projection 1 of {nprojections}")

    proj_T = snapGas.get_Aslice(
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

    proj_dens = snapGas.get_Aslice(
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

    proj_vol = snapGas.get_Aslice(
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
):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    zlimDict = copy.deepcopy(xlimDict)


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




if __name__ == "__main__":
    print(loadpath)
    rotation_matrix = None
    for snapNumber in snapRange:
        # snapNumber = 100
        print(f"[@{int(snapNumber)}]: Load subfind")
        # load in the subfind group files
        snap_subfind = load_subfind(snapNumber, dir=loadpath)

        print(f"[@{int(snapNumber)}]: Load snapshot")
        snapGas = gadget_readsnap(
            snapNumber,
            loadpath,
            hdf5=True,
            loadonlytype=[0, 1, 4],
            lazy_load=False,
            # subfind=snap_subfind,
        )

        print(f"[@{int(snapNumber)}]: Rotate and centre snapshot")
        snapGas.calc_sf_indizes(snap_subfind, halolist=[0])
        if rotation_matrix is None:
            print(f"[@{int(snapNumber)}]: New rotation of snapshots")
            rotation_matrix = snapGas.select_halo(snap_subfind, do_rotation=True)
        else:
            print(f"[@{int(snapNumber)}]: Existing rotation of snapshots")
            snapGas.select_halo(snap_subfind, do_rotation=False)
            snapGas.rotateto(
                rotation_matrix[0], dir2=rotation_matrix[1], dir3=rotation_matrix[2]
            )


        print(
            f"[@{int(snapNumber)}]: SnapShot loaded at RedShift z={snapGas.redshift:0.05e}"
        )


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
            # logParameters = logParameters,
            paramsOfInterest=["R","T","Tdens","rho_rhomean","n_H","gz","tcool","theat"],
            mappingBool=True,
            box=box,
            numthreads=numthreads,
            verbose = False,
        )

        print(
            f"[@{int(snapNumber)}]: Remove dark matter..."
        )

        whereDM = snapGas.data["type"] == 1

        snapGas = remove_selection(
            snapGas,
            removalConditionMask = whereDM,
            errorString = "Remove DM from Gas",
            DEBUG = DEBUG,
            )


        print(
            f"[@{int(snapNumber)}]: Remove stars..."
        )
        whereStars = snapGas.data["type"] == 4
        snapGas = remove_selection(
            snapGas,
            removalConditionMask = whereStars,
            errorString = "Remove Stars from Gas",
            DEBUG = DEBUG
            )

        #
        # print(
        #     f"[@{int(snapNumber)}]: Slice plot"
        # )
        #
        # plot_slices(snapGas,
        #     snapNumber,
        #     boxsize=Rvir*2.0*1.40
        # )
        #
        # print(
        #     f"[@{int(snapNumber)}]: Slice plot Quad"
        # )
        #
        # plot_slices_quad(snapGas,
        #     snapNumber,
        #     boxsize=Rvir*2.0*1.40
        # )
        #
        # print(
        #     f"[@{int(snapNumber)}]: Projection plot"
        # )
        #
        # plot_projections(snapGas,
        #     snapNumber,
        #     boxsize=Rvir*2.0*1.40
        # )

        print(
            f"[@{int(snapNumber)}]: Remove beyond Virial Radius..."
        )

        whereOutsideVirial = snapGas.data["R"] > Rvir

        xlimDict["R"]["xmax"] = Rvir

        snaptmp = remove_selection(
            snapGas,
            removalConditionMask = whereOutsideVirial,
            errorString = "Remove Outside Virial from Gas",
            DEBUG = DEBUG,
            )

        print(
            f"[@{int(snapNumber)}]: Convert from SnapShot to Dictionary and Trim ..."
        )
        # Make normal dictionary form of snapGas
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
            xParams = ["rho_rhomean","R","vol","tcool","theat"],
            weightKeys = ["mass","n_H","vol","gz","tcool","theat"]
        )

        print(
            f"[@{int(snapNumber)}]: Done"
        )
    print(loadpath)
