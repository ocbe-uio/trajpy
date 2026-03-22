Synthetic Data Generator
========================

TrajPy provides a built-in synthetic trajectory generator (``trajpy.traj_generator``) that can produce
four physically motivated types of particle motion. These synthetic trajectories are the foundation for
building labelled training datasets for machine-learning-based trajectory classification.

.. contents:: Contents
   :local:
   :depth: 2

----

Overview
--------

The module ``trajpy.traj_generator`` exposes one generator function for each diffusion regime and a
convenience helper to persist the results to disk:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Function
     - Description
   * - :func:`anomalous_diffusion`
     - Subdiffusion / superdiffusion via the Weierstrass–Mandelbrot function
   * - :func:`normal_diffusion`
     - Brownian (Fickian) diffusion via a Monte-Carlo acceptance–rejection scheme
   * - :func:`confined_diffusion`
     - Diffusion restricted to a circular confinement region
   * - :func:`superdiffusion`
     - Directed (ballistic) motion at constant velocity
   * - :func:`save_to_file`
     - Write trajectory arrays to CSV files

All generator functions share the same return convention:

.. code-block:: python

   x, y = generator(...)
   # x – 1-D array of time points,  shape (n_steps,)
   # y – position array
   #     shape (n_steps,)          when n_samples == 1
   #     shape (n_steps, n_samples) when n_samples  > 1

----

Diffusion Regimes
-----------------

Anomalous Diffusion
~~~~~~~~~~~~~~~~~~~

Anomalous diffusion encompasses both **subdiffusion** (:math:`\alpha < 1`) and
**superdiffusion** (:math:`\alpha > 1`). The mean squared displacement of an anomalous
process scales as:

.. math::

   \langle r^2(t) \rangle \propto t^{\alpha}

Trajectories are generated using the **Weierstrass–Mandelbrot** (WM) stochastic
function:

.. math::

   W(t) = \sum_{n=-\infty}^{\infty}
          \frac{\cos(\phi_n) - \cos(\gamma^n t^* + \phi_n)}
               {\gamma^{n\alpha/2}}

where :math:`\gamma = \sqrt{\pi}`, :math:`t^* = 2\pi t / N`, and :math:`\phi_n` is a
uniformly distributed random phase in :math:`[0, 2\pi)`.

.. function:: anomalous_diffusion(n_steps, n_samples, time_step, alpha)

   Generate an ensemble of anomalous diffusion trajectories.

   :param int n_steps: Number of time steps per trajectory.
   :param int n_samples: Number of independent trajectories to generate.
   :param float time_step: Duration of each time step :math:`\Delta t`.
   :param float alpha: Anomalous exponent (:math:`0 < \alpha < 2`).
                       Values below 1 produce subdiffusion; values above 1 produce superdiffusion.
   :returns: ``(x, y)`` – time array and position array.
   :rtype: tuple[numpy.ndarray, numpy.ndarray]

Normal Diffusion
~~~~~~~~~~~~~~~~

Normal (Brownian) diffusion produces trajectories whose mean squared displacement grows
linearly with time:

.. math::

   \langle r^2(t) \rangle = 4 D t

Steps are drawn using a Monte-Carlo **acceptance–rejection** method with the radial
probability density:

.. math::

   p(u) = \frac{2u}{4Dt} \exp\!\left(-\frac{u^2}{4Dt}\right)

where :math:`u` is the magnitude of the proposed displacement, :math:`D` is the diffusion
coefficient, and :math:`\Delta t` is the time step.

.. function:: normal_diffusion(n_steps, n_samples, dx, y0, D, dt)

   Generate an ensemble of normal diffusion trajectories.

   :param int n_steps: Number of time steps per trajectory.
   :param int n_samples: Number of independent trajectories.
   :param float dx: Maximum proposed step length (defines the proposal interval
                    :math:`[-dx/2,\, dx/2]`).
   :param float y0: Initial position.
   :param float D: Diffusion coefficient.
   :param float dt: Time step :math:`\Delta t`.
   :returns: ``(x, y)`` – time array and position array.
   :rtype: tuple[numpy.ndarray, numpy.ndarray]

Confined Diffusion
~~~~~~~~~~~~~~~~~~

Confined diffusion models a particle undergoing Brownian motion within a bounded region
of radius :math:`R`. At each macroscopic time step, a short normal-diffusion sub-trajectory
is simulated; only displacements that keep the particle inside the confinement region are
accepted.

.. function:: confined_diffusion(radius, n_steps, n_samples, dx, y0, D, dt)

   Generate trajectories under spatial confinement.

   :param float radius: Confinement radius :math:`R`.
   :param int n_steps: Number of time steps per trajectory.
   :param int n_samples: Number of independent trajectories.
   :param float dx: Maximum step length passed to the internal normal-diffusion sampler.
   :param float y0: Initial position.
   :param float D: Diffusion coefficient.
   :param float dt: Time step :math:`\Delta t`.
   :returns: ``(x, y)`` – time array and position array.
   :rtype: tuple[numpy.ndarray, numpy.ndarray]

Superdiffusion (Directed Motion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Superdiffusion via directed (ballistic) motion models a particle moving at constant
velocity :math:`v`, such that:

.. math::

   y(t + \Delta t) = y(t) + v \,\Delta t

This represents the fastest possible transport regime and is typically combined
pairwise with normal diffusion components to create realistic active-motion trajectories.

.. function:: superdiffusion(velocity, n_steps, n_samples, y0, dt)

   Generate directed (ballistic) motion trajectories.

   :param float velocity: Constant drift velocity :math:`v`.
   :param int n_steps: Number of time steps per trajectory.
   :param int n_samples: Number of independent trajectories.
   :param float y0: Initial position.
   :param float dt: Time step :math:`\Delta t`.
   :returns: ``(x, y)`` – time array and position array.
   :rtype: tuple[numpy.ndarray, numpy.ndarray]

----

Saving Trajectories to Disk
----------------------------

.. function:: save_to_file(y, param, path)

   Save a trajectory array to a CSV file.

   The output filename follows the pattern ``<path>/traj_<param>.csv`` and contains a
   header row ``t,x,y,...`` compatible with the :class:`trajpy.Trajectory` loader.

   :param numpy.ndarray y: Trajectory array of shape ``(n_steps, n_dims)`` or
                           ``(n_steps,)`` for a single 1-D trajectory.
   :param param: A scalar or string that characterises the trajectory (e.g. the value of
                 :math:`\alpha` or :math:`D`). Used in the filename.
   :type param: int | float | str
   :param str path: Directory where the file will be written.

----

Usage Examples
--------------

Anomalous Diffusion
~~~~~~~~~~~~~~~~~~~

Generate 20 anomalous trajectories spanning :math:`\alpha \in [0.1, 2.1]` and save each one:

.. code-block:: python

   import numpy as np
   import trajpy.traj_generator as tjg

   n_steps  = 250   # time steps per trajectory
   n_samples = 1    # one trajectory per alpha value
   dt        = 1.0  # time increment

   alphas = np.linspace(0.10, 2.1, 20)

   for alpha in alphas:
       x, y = tjg.anomalous_diffusion(n_steps, n_samples, dt, alpha=alpha)
       tjg.save_to_file(y, alpha, 'data/anomalous')

Normal Diffusion
~~~~~~~~~~~~~~~~

Generate trajectories for several diffusion coefficients:

.. code-block:: python

   import numpy as np
   import trajpy.traj_generator as tjg

   n_steps   = 250
   n_samples = 1
   dt        = 1.0
   diffusivity = np.array([10., 100., 1000., 10000.])

   for D in diffusivity:
       x, y = tjg.normal_diffusion(n_steps, n_samples, dx=1.0, y0=0., D=D, dt=dt)
       tjg.save_to_file(y, D, 'data/normal')

Confined Diffusion
~~~~~~~~~~~~~~~~~~

Generate confined trajectories for three confinement radii:

.. code-block:: python

   import numpy as np
   import trajpy.traj_generator as tjg

   n_steps   = 250
   n_samples = 1
   dt        = 1.0
   D         = 100.
   radii     = np.array([5., 10., 20.])

   for R in radii:
       x, y = tjg.confined_diffusion(R, n_steps, n_samples, dx=1.0, y0=0.0, D=D, dt=dt)
       tjg.save_to_file(y, R, 'data/confined')

Superdiffusion (Directed Motion)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate directed-motion trajectories for several velocities:

.. code-block:: python

   import numpy as np
   import trajpy.traj_generator as tjg

   n_steps   = 250
   n_samples = 1
   dt        = 1.0
   velocities = np.array([0.1, 1., 2., 5.])

   for v in velocities:
       x, y = tjg.superdiffusion(v, n_steps, n_samples, y0=0., dt=dt)
       tjg.save_to_file(y, v, 'data/superdiff')

----

Parameter Guidelines
--------------------

The table below lists recommended starting values for each trajectory type.

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 20 15

   * - Regime
     - ``n_steps``
     - ``dt``
     - Key parameter
     - Suggested range
   * - Anomalous
     - ≥ 100
     - 1.0
     - ``alpha``
     - 0.1 – 2.0
   * - Normal diffusion
     - ≥ 100
     - 1.0
     - ``D``
     - 10 – 10 000
   * - Confined
     - ≥ 100
     - 1.0
     - ``radius``
     - 5 – 50
   * - Superdiffusion
     - ≥ 100
     - 1.0
     - ``velocity``
     - 0.1 – 10

.. tip::
   Use ``n_samples > 1`` to produce an ensemble in a single call. The returned ``y``
   array will have shape ``(n_steps, n_samples)``, which can be iterated column-wise
   when saving individual trajectories.

----

API Reference
-------------

.. automodule:: trajpy.traj_generator
   :members:
   :undoc-members:
   :show-inheritance:

