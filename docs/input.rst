.. _input:

Input parameters
================

`beam_me_up() <scotty.beam_me_up.beam_me_up>` takes all the inputs of a run
as keyword arguments. `get_parameters_for_Scotty()
<scotty.init_bruv.get_parameters_for_Scotty>` gives them for the presets of
several diagnostics, as a dict to change and pass on::

    from scotty import beam_me_up, get_parameters_for_Scotty

    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    kwargs["launch_freq_GHz"] = 60.0
    dt = beam_me_up(**kwargs)

The magnetic field comes from ``find_B_method``: a field object (for
example `scotty.geometry.EFITField.from_MAST_saved`), a TORBEAM topfile
(``"torbeam"``, with the path of the file in ``magnetic_data_path``), or
``"analytical"``. The density comes from ``density_fit_method``: a profile
object from `scotty.profile_fit`, the name of a fit with
``density_fit_parameters``, or ``"smoothing-spline-file"`` with the path of
the file in ``ne_data_path``.

A script written for Scotty v3 runs with
``beam_me_up(**scotty.compat.from_v3_kwargs(kwargs))`` (see
`scotty.compat`).

The full list of arguments is the documentation of `beam_me_up`:

.. autofunction:: scotty.beam_me_up.beam_me_up
   :noindex:
