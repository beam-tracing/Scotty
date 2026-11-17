Scotty
======

Scotty is a beam-tracing code for Doppler backscattering (DBS) synthetic
diagnostics in tokamaks. It launches a Gaussian beam, traces it through the
plasma, and computes where along the beam the signal comes from and how
strong it is: the filter functions of the beam model of DBS [[1]](#1).


Status
------

- **v3 is the stable version.** The latest release is 3.2.0. It works in
  cylindrical coordinates (R, ζ, Z). Use v3 for published work, or for
  anything that you must reproduce later.
- **v4 is at its first release candidate, 4.0.0rc1**, on the `master` branch.
  It has the same beam model, with cylindrical or Cartesian (X, Y, Z)
  coordinates (`geometry="cylindrical"` or `"cartesian"`), and it gives its
  outputs in both. The rc freezes the interface (the argument names and
  defaults, what the inputs mean, and the output variables), so scripts
  written for it keep working in 4.0.0. It has been checked against v3 on the
  test suite and on DIII-D, MAST-U and MAST cases. Please try it, and report
  problems on the [issue tracker](https://github.com/beam-tracing/Scotty/issues).

Check which version you have with `scotty.__version__`. An install from
`master` after the rc gives a development version such as
`4.0.0rc2.dev5+g1234abc` (from `setuptools_scm`).


Install
-------

The package is called `scotty-beam-tracing`, and the Python module is called
`scotty`. Scotty needs Python 3.10 or newer.

- v3, from PyPI:

  ```
  pip install scotty-beam-tracing
  ```

- v4, the release candidate, from PyPI (pip installs a release candidate
  only when you ask for it):

  ```
  pip install scotty-beam-tracing==4.0.0rc1
  ```

- v4, the latest development version, from GitHub:

  ```
  pip install git+https://github.com/beam-tracing/Scotty
  ```

- To develop Scotty, from a local clone:

  ```
  pip install -e ".[tests]"
  ```


Quick start (v4)
----------------

```python
from scotty import beam_me_up, get_parameters_for_Scotty

# A DBS beam in an analytical field with circular flux surfaces
kwargs = get_parameters_for_Scotty("DBS_synthetic")
dt = beam_me_up(**kwargs)  # also saved as scotty_output.h5
print(dt["diagnostics/dbs"]["filterfun_all"])
```

More examples are in [`Examples/`](Examples/).


Moving from v3 to v4
--------------------

- A v3 script runs with `beam_me_up(**scotty.compat.from_v3_kwargs(kwargs))`.
  It converts v3's arguments and logs each change.
- `find_B_method` and `density_fit_method` must be given: they have no
  defaults.
- The data files are given by their paths (`magnetic_data_path`,
  `ne_data_path`, `Te_data_path`). EFIT and saved UDA files are read with the
  field constructors in `scotty.geometry`, and the field is given to
  `beam_me_up` as `find_B_method`.
- The output has the groups `inputs`, `solver`, `beam` and `diagnostics/dbs`.
  `scotty.compat.convert_output_file` converts output files between the two
  versions.
- The release notes list every change, and say which ones change results:
  [`docs/release_notes.rst`](docs/release_notes.rst).


Documentation
-------------

https://scotty.readthedocs.io/: the inputs, the output, the layout of the code,
the API and the release notes.


Contribute
----------

- Source code: https://github.com/beam-tracing/Scotty
- Issue tracker: https://github.com/beam-tracing/Scotty/issues
- Run the tests with `pytest`, and format the code with `black` (CI checks
  both). Open pull requests against `master`.


Get help
--------

Open an issue on the [issue tracker](https://github.com/beam-tracing/Scotty/issues).


Cite
----

If you use Scotty in published work, cite [[1]](#1). It describes the beam
model, and its Appendix A derives beam tracing.

<a id="1">[1]</a>
Hall-Chen, V. H., Parra, F. I., & Hillesheim, J. C. (2022).
[Beam model of Doppler backscattering](https://iopscience.iop.org/article/10.1088/1361-6587/ac57a1).
*Plasma Physics and Controlled Fusion*, **64**(9), 095002.


License
-------

Scotty is licensed under the GNU General Public License, version 3
([`LICENSE.md`](LICENSE.md)).
