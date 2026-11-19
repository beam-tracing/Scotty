Scotty
======

Scotty is a beam-tracing code for Doppler backscattering (DBS) synthetic
diagnostics in tokamaks. It launches a Gaussian beam, traces it through the
plasma, and computes where along the beam the signal comes from and how
strong it is: the filter functions of the beam model of DBS (see
:ref:`references`).

To run it, call `beam_me_up() <scotty.beam_me_up.beam_me_up>`.
`get_parameters_for_Scotty() <scotty.init_bruv.get_parameters_for_Scotty>`
gives the inputs for several diagnostics:

.. code-block:: python

    from scotty import beam_me_up, get_parameters_for_Scotty

    kwargs = get_parameters_for_Scotty("DBS_synthetic")
    dt = beam_me_up(**kwargs)

Install it with ``pip install scotty-beam-tracing`` (v3, the stable
version) or ``pip install git+https://github.com/beam-tracing/Scotty`` (v4,
in development). The source code and the issue tracker are on
`GitHub <https://github.com/beam-tracing/Scotty>`_.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   input
   output
   naming
   code_layout
   references
   release_notes
   api/modules


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
