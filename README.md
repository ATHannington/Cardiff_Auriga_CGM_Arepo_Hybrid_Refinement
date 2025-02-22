# Hybrid Refinement methods in Arepo - analysis tools
### Forming the basis of the research into: the development of two levels of resolution ("hybrid") methods, and the identification and assessment of performance, accuracy, and any numerical artefacts. Developed in Arepo, as applied to simulations of the multi-phase Circumgalactic Medium (CGM) of low-redshift Milky Way-mass Galaxies.
### Using cosmological zoom-in Auriga galaxy formation and evolution simulations using the massively parallelised magneto-hydrodynamical simulation code Arepo.

The analysis tools in this repository were developed to build upon the `Tracers_Subroutines.py` package of my previous work in [CGM_Multi-Phase](https://github.com/ATHannington/Cardiff_Auriga_CGM_Multi_Phase). 
Moreover, the tools of this repository were developed in conjunction with `CR_Subroutines.py` [CGM_Cosmic-Rays](https://github.com/ATHannington/Cardiff_Auriga_CGM_Cosmic_Rays). 
Both `Tracers_Subroutines.py` and `CR_Subroutines.py` are regularly updated with the latest versions from their respective repos, but for each of these, the versions presented in this repo should be considered the latest stable version.
In this manner, the repositories are cross-dependent but can be used as standalone toolsets.

---

## Affiliations
*Author*: Andrew Tomos Hannington

*Affiliation*: Cardiff University

*Supervisor*: Dr Freeke van de Voort (*Email*: vandevoortF@cardiff.ac.uk)

> [!IMPORTANT]
> The code in this repository builds upon proprietary software from the Auriga Project (https://wwwmpa.mpa-garching.mpg.de/auriga/) collaboration that is not available to the general public (there is, however, a publicly available version of Arepo here: https://arepo-code.org/wp-content/userguide/index.html). As such, **the code of this repository cannot be run without membership of the Auriga Project collaboration and access to Arepo Snap Utils**. Thus, for all other persons, the software of this repository is for illustrative purposes only.

## The introduction in brief
TO DO

## Software outline
TO DO
---

## Final notes
Wherever code is commented out in the versions presented here they have been left in for my own future reference, or for ease of debugging and development. For example, it saves significant time to be able to uncomment out a few dummy function calls needed to obtain the necessary data to debug sebsequent functions, rather than to write these dummy function calls from scratch each time (they are frequently unique, subtle modifications to similar function calls made elsewhere in the code, and remembering exactly which variant is needed at any given line of code is quite difficult). There are also entire functions and old versions commented out too. Again, these are for my ease of reference only.
