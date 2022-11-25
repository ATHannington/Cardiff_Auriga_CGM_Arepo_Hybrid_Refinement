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
snapEnd = 109#116
DEBUG = False
forceLogMass = False
numthreads = 18

loadPathBase = "/home/cosmos/c1838736/Auriga/level5_cgm/"
loadDirectories = [
    # "high-time-resolution/h5_1kpc_snapshot-restart-of-2kpc",
    # "high-time-resolution/h5_2kpc_snapshot-restart-of-2kpc",
    # "high-time-resolution/h5_hy-v2_snapshot-restart-of-2kpc",
    # "h5_standard",
    # "h5_2kpc",
    # "snapshot-restart-of-2kpc/h5_1kpc_snapshot-restart-of-2kpc",
    # "snapshot-restart-of-2kpc/h5_hy-v2_snapshot-restart-of-2kpc",
    "h5_standard",
    "h5_1kpc",
    "snapshot-restart-of-1kpc/h5_hy-v2_snapshot-restart-of-1kpc",
]

residualsReferenceSimDict = {
    "high-time-resolution/h5_1kpc_snapshot-restart-of-2kpc": "high-time-resolution/h5_2kpc_snapshot-restart-of-2kpc",
    "high-time-resolution/h5_2kpc_snapshot-restart-of-2kpc": None,
    "high-time-resolution/h5_hy-v2_snapshot-restart-of-2kpc": "high-time-resolution/h5_2kpc_snapshot-restart-of-2kpc",
    "h5_standard": None,
    "h5_2kpc": None,
    "snapshot-restart-of-2kpc/h5_1kpc_snapshot-restart-of-2kpc": "h5_2kpc",
    "snapshot-restart-of-2kpc/h5_hy-v2_snapshot-restart-of-2kpc": "h5_2kpc",
    "h5_standard": None,
    "h5_1kpc": None,
    "snapshot-restart-of-1kpc/h5_hy-v2_snapshot-restart-of-1kpc": "h5_1kpc",
}

simulations = []
savePaths = []

tmp = {}
for dir in loadDirectories:
    loadpath = loadPathBase+dir+"/output/"
    simulations.append(loadpath)

    savepath = "./" + dir + "/"
    savePaths.append(savepath)
    newKey = savepath
    if residualsReferenceSimDict[dir] is not None:
        newReferencePath = "./" + copy.deepcopy(residualsReferenceSimDict[dir]) + "/"
    else:
        newReferencePath = None
    tmp.update({newReferencePath : newReferencePath})

residualsReferenceSimDict = tmp

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


def combined_pdf_versus_plot(
    savePaths,
    ylabel,
    xlimDict,
    logParameters,
    snapNumber,
    weightKeys = ['mass'],
    xParams = ["T"],
    titleBool=False,
    DPI=150,
    xsize=8.0,
    ysize=8.0,
    fontsize=13,
    fontsizeTitle=14,
    cumulative = False,
    SFR = False,
    byType = False,
    colourmapMain = "plasma",
    forceLogMass = False,
    residualsReferenceSimDict = None,
):

    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    try:
        tmp = xlimDict["mass"]
    except:
        xlimDict.update({"mass": {"xmin": 4.0, "xmax": 9.0}})

    limDict = copy.deepcopy(xlimDict)

    if SFR is True:
        weightKeys = ["gima"]
        xParams = ["age"]

    if byType is True:
        uniqueTypes = [0,1,2,3,4,5]
        for tp in uniqueTypes:
            print("Starting type ",tp)

            tmpSavePaths = []
            for savePath in savePaths:
                path = savePath + f"type{int(tp)}/"
                tmpSavePaths.append(path)

            combined_pdf_versus_plot(
                savePaths = tmpSavePaths,
                ylabel = ylabel,
                xlimDict = xlimDict,
                logParameters = logParameters,
                snapNumber = snapNumber,
                weightKeys = weightKeys,
                xParams = xParams,
                titleBool = titleBool,
                DPI = DPI,
                xsize = xsize,
                ysize = ysize,
                fontsize = fontsize,
                fontsizeTitle = fontsizeTitle,
                cumulative = cumulative,
                SFR = SFR,
                byType = False,
                colourmapMain = colourmapMain,
                forceLogMass = forceLogMass,
                residualsReferenceSimDict = residualsReferenceSimDict,
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
            Nsims = len(savePaths)
            for (ii,savePathBase) in enumerate(savePaths):
                print("savePathBase= ",savePathBase)

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


                SFRBool = False
                if (weightKey == "gima")&(analysisParam=="age"):
                    SFRBool = True

                if cumulative is True:
                    tmp2 = savePath +"Cumulative-"
                else:
                    tmp2 = savePath

                if SFRBool is True:
                    opslaan = tmp2 + f"SFR_{snapNumber}"
                else:
                    opslaan = tmp2 + f"{weightKey}-{analysisParam}-PDF_{snapNumber}"

                print(opslaan)

                # out = {"data":{"x" : xFromBins, "y" : hist}}
                try:
                    dataDict = hdf5_load(opslaan+"_data.h5")
                except:
                    print(f"Load path of {opslaan+'_data.h5'} not found! Skipping...")
                    continue
                #
                # referenceSimPath = residualsReferenceSimDict[savePathBase]
                # print("savePathBase= ",savePathBase)
                #
                # savePath = savePathBase + "Plots/PDFs/"
                # tmp = "./"
                #
                # for savePathChunk in savePath.split("/")[1:-1]:
                #     tmp += savePathChunk + "/"
                #     try:
                #         os.mkdir(tmp)
                #     except:
                #         pass
                #     else:
                #         pass
                #
                #
                # SFRBool = False
                # if (weightKey == "gima")&(analysisParam=="age"):
                #     SFRBool = True
                #
                # if cumulative is True:
                #     tmp2 = savePath +"Cumulative-"
                # else:
                #     tmp2 = savePath
                #
                # if SFRBool is True:
                #     opslaan = tmp2 + f"SFR_{snapNumber}"
                # else:
                #     opslaan = tmp2 + f"{weightKey}-{analysisParam}-PDF_{snapNumber}"
                #
                # print(opslaan)
                #
                # # out = {"data":{"x" : xFromBins, "y" : hist}}
                # try:
                #     dataDict = hdf5_load(opslaan+"_data.h5")
                # except:
                #     print(f"Load path of {opslaan+'_data.h5'} not found! Skipping...")
                #     continue



                cmap = matplotlib.cm.get_cmap(colourmapMain)
                if colourmapMain == "tab10":
                    colour = cmap(float(ii) / 10.0)
                else:
                    colour = cmap(float(ii) / float(Nsims))

                splitbase = savePathBase.split("/")
                # print(splitbase)
                if "" in splitbase:
                    splitbase.remove("")
                if "." in splitbase:
                    splitbase.remove(".")
                # print(splitbase)

                if len(splitbase)>2:
                    label = f'{splitbase[0]}: {"_".join(((splitbase[-2]).split("_"))[:2])} ({splitbase[-1]})'
                elif len(splitbase)>1:
                    label = f'{splitbase[0]}: {"_".join(((splitbase[-1]).split("_"))[:2])}'
                else:
                    label = f'Original: {"_".join(((splitbase[-1]).split("_"))[:2])}'
                # print("label= ",label)
                ax.plot(
                    dataDict["data"]["x"],
                    dataDict["data"]["y"],
                    color=colour,
                    linestyle="solid",
                    label = label
                )

                ax.xaxis.set_minor_locator(AutoMinorLocator())
                ax.yaxis.set_minor_locator(AutoMinorLocator())
                ax.tick_params(
                    axis="both", which="both", labelsize=fontsize
                )

            ylabel_prefix = ""
            if cumulative is True:
                ylabel_prefix = "Cumulative "
            if weightKey == "mass":
                if forceLogMass is False:
                    ax.set_ylabel(ylabel_prefix+r"Mass (M$_{\odot}$)", fontsize=fontsize)
                else:
                    ax.set_ylabel(r"$Log_{10}$ "+ylabel_prefix+"Mass (M$_{\odot}$)", fontsize=fontsize)
            else:
                ax.set_ylabel(
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
            ax.legend(loc="upper left", fontsize=fontsize)

            # plt.tight_layout()
            if titleBool is True:
                plt.subplots_adjust(top=0.875, hspace=0.1, left=0.15)
            else:
                plt.subplots_adjust(hspace=0.1, left=0.15)

            savePath = "./Combined-Plots/" + opslaan[2:]
            tmp = "./"

            for savePathChunk in savePath.split("/")[1:-1]:
                tmp += savePathChunk + "/"
                try:
                    os.mkdir(tmp)
                except:
                    pass
                else:
                    pass

            plt.savefig(savePath + "_sims-combined" + ".pdf", dpi=DPI, transparent=False)
            print("Saved as: ",savePath + "_sims-combined" + ".pdf")
            plt.close()
    return


if __name__ == "__main__":
    for snapNumber in snapRange:
        print(
            f"[@{int(snapNumber)}]: PDF of mass vs R plot..."
        )

        combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["R"],
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: Cumulative PDF of mass vs R plot..."
        )

        combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["R"],
            cumulative = True,
            forceLogMass = forceLogMass,

        )

        print(
            f"[@{int(snapNumber)}]: By Type PDF of mass vs R plot..."
        )

        combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["R"],
            byType = True,
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: By Type Cumulative PDF of mass vs R plot..."
        )

        combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["R"],
            cumulative = True,
            byType = True,
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: SFR plot..."
        )

        combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['gima'],
            xParams = ["age"],
            SFR = True,
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: Cumulative SFR plot..."
        )

        combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['gima'],
            xParams = ["age"],
            cumulative = True,
            SFR = True,
            forceLogMass = forceLogMass,
        )

        print(
            f"[@{int(snapNumber)}]: PDF of gas (mass vs T or vol) plot"
        )


        combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["T","vol"],
            forceLogMass = forceLogMass,
        )


        print(
            f"[@{int(snapNumber)}]: Cumulative PDF of gas (mass vs T or vol) plot"

        )
        combined_pdf_versus_plot(
            savePaths,
            ylabel,
            xlimDict,
            logParameters,
            snapNumber,
            weightKeys = ['mass'],
            xParams = ["T","vol"],
            cumulative = True,
            forceLogMass = forceLogMass,
        )

    print("***")
    print("...done!")
    print("***")
