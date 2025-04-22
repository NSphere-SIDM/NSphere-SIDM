.. _command_line_usage:

Command-Line Usage
==================

This section details the command-line options for the executable components
of the NSphere simulation and analysis suite. All executables should be run
from the main project directory.

.. contents:: Table of Contents
   :local:
   :depth: 1

N-Spherical Shell Simulation (`nsphere`)
---------------------------------------

This is the core C program that runs the N-spherical shell simulation.

.. program:: nsphere

**Usage:**

.. code-block:: bash

   ./nsphere [options]

**Options:**

.. option:: --help

   Show the usage message and exit.

.. option:: --restart

   Enable restart mode. Looks for existing output data products (primarily
   ``all_particle_data<suffix>.dat`` and snapshot files) to determine where
   to resume processing. If complete data is found for the simulation phase,
   it may be skipped entirely. Incompatible with ``--readinit`` and ``--writeinit``.
   [Default: Off]

.. option:: --nparticles <int>

   Number of particles to simulate (after potential tidal stripping).
   [Default: 100000]

.. option:: --ntimesteps <int>

   Requested total number of simulation timesteps. Note: This value may be
   adjusted slightly upwards by the program to ensure alignment with the
   snapshot output schedule defined by ``--nout`` and ``--dtwrite``.
   [Default: 10000]

.. option:: --nout <int>

   Number of desired output snapshot intervals (results in ``nout + 1`` actual
   snapshots, including t=0).
   [Default: 100]

.. option:: --dtwrite <int>

   Number of simulation timesteps between each major data write/snapshot interval.
   [Default: 100]

.. option:: --tag <string>

   Append a custom string tag to the automatically generated suffix for most
   output filenames. Useful for distinguishing different runs with the same
   core parameters.
   [Default: None]

.. option:: --method <int>

   Selects the integration algorithm. [Default: 1]

   1
     : Selects the adaptive Leapfrog integrator combined with adaptive Levi-Civita regularization for close encounters (Default).
   2
     : Selects the full-step adaptive Leapfrog integrator combined with Levi-Civita regularization.
   3
     : Selects the full-step adaptive Leapfrog integrator without regularization.
   4
     : Selects the 4th-order symplectic Yoshida integrator.
   5
     : Selects the 3rd-order Adams-Bashforth predictor-corrector method.
   6
     : Selects the standard Leapfrog integrator with the velocity half-step formulation (Kick-Drift-Kick).
   7
     : Selects the standard Leapfrog integrator with the position half-step formulation (Drift-Kick-Drift).
   8
     : Selects the classic 4th-order Runge-Kutta integrator.
   9
     : Selects the simple forward Euler integration method.

.. option:: --methodtag

   Include the integration method name string (e.g., "adp.leap.adp.levi")
   in the output filename suffix, in addition to the parameters.
   [Default: Off]

.. option:: --sort <int>

   Selects the particle sorting algorithm. [Default: 1]

   1
     : Selects Parallel Quadsort (Default).
   2
     : Selects Sequential Quadsort.
   3
     : Selects Parallel Insertion Sort.
   4
     : Selects Sequential Insertion Sort.

.. option:: --readinit <file>

   Read initial particle conditions (positions, velocities, etc.) directly
   from the specified binary ``<file>`` located inside the ``init/`` subdirectory,
   instead of generating them. The file must have been created previously
   using ``--writeinit``. Incompatible with ``--restart`` and ``--writeinit``.
   [Default: Off]

.. option:: --writeinit <file>

   Generate initial particle conditions and save them to the specified binary
   ``<file>`` inside the ``init/`` subdirectory. The simulation then
   proceeds normally. Incompatible with ``--restart`` and ``--readinit``.
   [Default: Off]

.. option:: --tfinal <int>

   Sets the total simulation duration as a multiple of the characteristic
   dynamical time (tdyn). Duration = ``<int>`` * tdyn.
   [Default: 5]

.. option:: --ftidal <float>

   Specifies the fraction of the outermost particles (by initial radius)
   to remove via tidal stripping before starting the simulation. Value must
   be between 0.0 (no stripping) and 1.0. The initial number of generated
   particles is increased to ensure ``--nparticles`` remain after stripping.
   [Default: 0.0]

.. option:: --save <subarg> [subarg...]

   Controls which major data products are saved. Can specify multiple sub-arguments;
   if conflicting levels are given, the one enabling the most output takes effect.
   [Default: all]

   raw-data
     : Saves only basic particle output files (`particles.dat`, `particlesfinal.dat`).
   psi-snaps
     : In addition to `raw-data`, saves potential snapshots (`Psi_methodA_t*.dat`) and enables dynamic Psi calculation.
   full-snaps
     : In addition to `psi-snaps` output, saves full snapshot data (`Rank_Mass_Rad_VRad_*.dat`) and enables dynamic rank calculation.
   debug-energy
     : Enables energy tracking mode. In addition to `full-snaps` output, saves a detailed energy diagnostic file (`debug_energy_compare.dat`) for particle tracking.
   all
     : Saves all possible outputs, equivalent to enabling `debug-energy`.

.. option:: --enable-log

   Enable detailed logging to ``log/nsphere.log``.
   [Default: Off]

Last Parameters File (`lastparams.dat`)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Upon successful startup, ``nsphere`` records the key parameters for the current run
into a suffixed file, typically named ``data/lastparams_<suffix>.dat`` (where
``<suffix>`` depends on the ``--tag``, ``--methodtag``, and core parameters like
``npts``, ``Ntimes``, ``tfinal``).

Crucially, it also creates or updates a standard file named ``data/lastparams.dat``
which acts as a symbolic link (on Unix-like systems) or a direct copy (on Windows)
of the most recently created suffixed parameter file.

The format of this file is a single line containing:
``<npts> <Ntimes> <tfinal_factor> [file_tag]``

This allows the ``nsphere_plot`` script (and related wrappers) to easily find
and use the parameters from the very last simulation run by default when no
``--suffix`` is specified.

Plotting & Animation (`nsphere_plot`)
------------------------------------

This Python script generates various plots and animations from the data
produced by the ``nsphere`` simulation.

.. program:: nsphere_plot

**Usage:**

.. code-block:: bash

   ./nsphere_plot [options]

**Options:**

.. option:: --suffix <SUFFIX>

   Specify the data file suffix (e.g., ``_tag_40000_1001_5``) used to find
   input data files generated by ``nsphere``.
   If omitted, the script attempts to read ``data/lastparams.dat`` to
   determine the suffix automatically from the parameters of the last ``nsphere``
   run. The ``lastparams.dat`` file is expected to contain a single line
   in the format: ``<npts> <Ntimes> <tfinal_factor> [file_tag]``.

.. option:: --start <N>

   Starting snapshot number for animations or time-series plots.
   [Default: 0]

.. option:: --end <N>

   Ending snapshot number for animations or time-series plots.
   A value of 0 means use all available snapshots up to the maximum found.
   [Default: 0]

.. option:: --step <N>

   Step size between snapshots used for animations.
   [Default: 1]

.. option:: --fps <N>

   Frames per second for output GIF animations.
   [Default: 10]

.. option:: --log

   Enable detailed logging to ``log/nsphere_plot.log``. Captures INFO and DEBUG
   messages in addition to warnings and errors.
   [Default: Off]

.. option:: --paced

   Run in "paced mode", adding artificial delays between major processing
   sections (e.g., between loading data and generating plots) for visual effect
   or to observe progress more slowly.
   [Default: Off]

.. option:: --help, -h

   Show the help message and exit.

**Visualization Control Flags:**

These flags control which types of plots or animations are generated. If none
of the ``--<type>`` flags (e.g., ``--animations``) are specified, the script
attempts to generate *all* available visualizations. If one or more ``--<type>``
flags *are* specified, *only* those types are generated.

*Run Only* Flags:

.. option:: --phase-space

   Generate only the initial phase space histogram and phase space animation.

.. option:: --phase-comparison

   Generate only the side-by-side initial vs final phase space plot and difference plot.

.. option:: --profile-plots

   Generate only the 1D profile plots (Density, Mass, Potential, f(E), etc.).

.. option:: --trajectory-plots

   Generate only particle trajectory plots and related diagnostics (Energy/Angular Momentum vs time).

.. option:: --2d-histograms

   Generate only the 2D phase space histograms (from particles*.dat and 2d_hist*.dat).

.. option:: --convergence-tests

   Generate only plots comparing results from different numerical parameters (Nintegration, Nspline).

.. option:: --animations

   Generate only the output GIF animations (Phase Space, Mass, Density, Psi).

.. option:: --energy-plots

   Generate only the Energy vs Time plots (including the debug comparison if data exists).

.. option:: --distributions

   Generate only the 1D comparison histograms of variable distributions (Radius, Velocity, etc.).

*Skip* Flags (used when running in default "all" mode):

.. option:: --no-phase-space

   Skip generating the initial phase space histogram and animation.

.. option:: --no-phase-comparison

   Skip generating the phase space comparison plots.

.. option:: --no-profile-plots

   Skip generating the 1D profile plots.

.. option:: --no-trajectory-plots

   Skip generating trajectory and related diagnostic plots.

.. option:: --no-histograms

   Skip generating all 2D histogram plots.

.. option:: --no-convergence-tests

   Skip generating the convergence test plots.

.. option:: --no-animations

   Skip generating all output GIF animations.

.. option:: --no-energy-plots

   Skip generating the Energy vs Time plots.

.. option:: --no-distributions

   Skip generating the 1D variable distribution comparison histograms.

Wrapper Scripts
~~~~~~~~~~~~~~~

These scripts provide convenient shortcuts to run specific parts of the main
``nsphere_plot`` script. They accept ``--suffix`` and ``--log`` arguments,
which are passed through to ``nsphere_plot``.

.. program:: nsphere_distributions

Generates 1D variable distribution comparison histograms (Initial vs Final).
Calls ``nsphere_plot --distributions``.

**Usage:**

.. code-block:: bash

   ./nsphere_distributions [--suffix SUFFIX] [--log]

**Options:**

.. option:: --suffix <SUFFIX>

   Data file suffix. Tries to read ``lastparams.dat`` if omitted.

.. option:: --log

   Enable detailed logging in ``nsphere_plot.py``.


.. program:: nsphere_2d_histograms

Generates 2D histogram plots (from particle files and nsphere.c output).
Calls ``nsphere_plot --2d-histograms``.

**Usage:**

.. code-block:: bash

   ./nsphere_2d_histograms [--suffix SUFFIX] [--log]

**Options:**

.. option:: --suffix <SUFFIX>

   Data file suffix. Tries to read ``lastparams.dat`` if omitted.

.. option:: --log

   Enable detailed logging in ``nsphere_plot.py``.


.. program:: nsphere_animations

Generates all standard animations (Phase Space, Mass, Density, Psi).
Calls ``nsphere_plot --animations``.

**Usage:**

.. code-block:: bash

   ./nsphere_animations [--suffix SUFFIX] [--start N] [--end N] [--step N] [--fps N] [--log]

**Options:**

.. option:: --suffix <SUFFIX>

   Data file suffix. Tries to read ``lastparams.dat`` if omitted.

.. option:: --start <N>

   Starting snapshot number. [Default: 0]

.. option:: --end <N>

   Ending snapshot number (0=auto). [Default: 0]

.. option:: --step <N>

   Snapshot step size. [Default: 1]

.. option:: --fps <N>

   Output animation frames per second. [Default: 10]

.. option:: --log

   Enable detailed logging in ``nsphere_plot.py``.