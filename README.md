Scotty
======

Beam tracing for tokamak DBS diagnostics


Scotty v3 and v4
----------------

**Scotty v4 is still unstable.** It is the version on the `master` branch, but it
hasn't been released yet. Its interface and outputs may still change between
commits, and some of v3's features haven't been ported. For published work, or
anything you need to reproduce later, use v3.

- **v3** (stable, the released version): beam tracing in cylindrical coordinates
  (R, ζ, Z). The latest release is 3.2.0.
- **v4** (in development): the same beam model, with a choice of cylindrical or
  Cartesian (X, Y, Z) coordinates (`geometry="cylindrical"` or `"cartesian"`).
  Outputs are given in both. It has been checked against v3 on the test suite
  and on DIII-D, MAST-U and MAST cases. Its known differences from v3:
  - some arguments and output names have changed. `scotty.compat.from_v3_kwargs`
    translates v3's `beam_me_up` arguments, and `scotty.compat.quick_cutoff`
    replaces v3's `quick_run`;
  - not yet ported: v3's plotting functions, the localisation widths (`delta_l`,
    `delta_kperp1`), and the `curvy_slab` field;
  - until v4 is tagged, its version number is a development version of 3.x (from
    `setuptools_scm`), so check which one you have with `scotty.__version__`.


Install
-------

The package to install is called `scotty-beam-tracing`, but the Python module is
just `scotty`.

**v3 (stable).** From PyPI:

```
pip install scotty-beam-tracing
```

or a specific release straight from GitHub:

```
pip install git+https://github.com/beam-tracing/Scotty@v3.2.0
```

**v4 (unstable).** The development version, straight from GitHub:

```
pip install git+https://github.com/beam-tracing/Scotty
```

or from a local clone:

```
pip install .
```

when developing, use:

```
pip install -e .
```

Documentation
-------
https://scotty.readthedocs.io/

Background
-------
Description of the Scotty [[1]](#1), derivation of beam tracing given in Appendix A.

## References
<a id="1">[1]</a> 
Hall-Chen, V. H., Parra, F. I., & Hillesheim, J. C. (2022). 
[Beam model of Doppler backscattering](https://iopscience.iop.org/article/10.1088/1361-6587/ac57a1). 
*Plasma Physics and Controlled Fusion*, **64**(9), 095002.
