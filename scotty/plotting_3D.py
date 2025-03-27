import matplotlib.pyplot as plt
import numpy as np
from scotty.typing import PathLike
from typing import Optional
import xarray as xr

def maybe_make_axis(ax: Optional[plt.Axes], *args, **kwargs) -> plt.Axes:
    if ax is None:
        _, ax = plt.subplots(*args, **kwargs)
    return ax

def plot_dispersion_relation(
    analysis: xr.Dataset,
    filename: Optional[PathLike] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    """
    Plots Cardano's and np.linalg's solutions to the actual dispersion relation
    Useful to check whether the solution which = 0 along the path changes
    """

    ax = maybe_make_axis(ax)

    ## The title of 'col' in H_eigvals is a misnomer
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_eigvals.sel(col="X")),
        color="pink",
        linestyle="-",
        linewidth=4,
        label="H_eigvals_1 (X?)",
    )
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_eigvals.sel(col="Y")),
        color="lightgreen",
        linestyle="-",
        linewidth=4,
        label="H_eigvals_2 (O?)",
    )
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_eigvals.sel(col="Z")),
        color="gray",
        linestyle="-",
        linewidth=4,
        label="H_eigvals_3",
    )

    ax.plot(
        analysis.l_lc,
        abs(analysis.H_1_Cardano),
        color="k",
        linestyle=":",
        linewidth=4,
        label="H_1_Cardano",
    )
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_2_Cardano),
        color="r",
        linestyle=":",
        linewidth=4,
        label="H_2_Cardano (X?)",
    )
    ax.plot(
        analysis.l_lc,
        abs(analysis.H_3_Cardano),
        color="g",
        linestyle=":",
        linewidth=4,
        label="H_3_Cardano (O?)",
    )

    ax.set_title("Dispersion relation")
    ax.legend()

    if filename:
        plt.savefig(f"{filename}.png")

    return ax