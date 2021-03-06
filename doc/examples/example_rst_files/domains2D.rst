.. _domains2D:

=======================================================================
Example: Generalized Chains with a 2,2-dimensional data,parameter space
=======================================================================

This example demonstrates the adaptive generation of samples using  a
goal-oriented adaptive sampling algorithm.

Generating a single set of adaptive samples
-------------------------------------------

We will walk through the following `example
<https://github.com/UT-CHG/BET/blob/master/examples/fromFile_ADCIRCMap/plotDomains2D.py>`_ that visualizes
samples on a regular grid for a 2-dimensional parameter space. 

The modules required by this example are::

    import numpy as np
    import scipy.io as sio
    import bet.postProcess.plotDomains as pDom

The compact (bounded, finite-dimensional) paramter space is::

    # [[min \lambda_1, max \lambda_1], [min \lambda_2, max \lambda_2]]
    lam_domain = np.array([[.07, .15], [.1, .2]])
    param_min = lam_domain[:, 0]
    param_max = lam_domain[:, 1]

In this example we form a linear interpolant to the QoI map :math:`Q(\lambda) =
(q_1(\lambda), q_6(\lambda))` using data read from a ``.mat`` :download:`file
<../../../examples/matfiles/Q_2D.mat>`::

    station_nums = [0, 5] # 1, 6
    mdat = sio.loadmat('Q_2D')
    Q = mdat['Q']
    Q = Q[:, station_nums]
    points = mdat['points']


Next, we implicty designate the region of interest :math:`\Lambda_k =
Q^{-1}(D_k)` in :math:`\Lambda` for some :math:`D_k \subset \mathcal{D}`
through the use of some kernel. In this instance we choose our kernel
:math:`p_k(Q) = \rho_\mathcal{D}(Q)`, see
:class:`~bet.sampling.adaptiveSampling.rhoD_kernel`.

We choose some :math:`\lambda_{ref}` and let :math:`Q_{ref} = Q(\lambda_{ref})`::

    Q_ref = mdat['Q_true']
    Q_ref = Q_ref[15, station_nums] # 16th/20

We define a rectangle, :math:`R_{ref} \subset \mathcal{D}` centered at
:math:`Q(\lambda_{ref})` with sides 15% the length of :math:`q_1` and
:math:`q_6`. Set :math:`\rho_\mathcal{D}(q) = \frac{\mathbf{1}_{R_{ref}}(q)}{||\mathbf{1}_{R_{ref}}||}`::

    bin_ratio = 0.15
    bin_size = (np.max(Q, 0)-np.min(Q, 0))*bin_ratio
    # Create kernel
    maximum = 1/np.product(bin_size)
    def rho_D(outputs):
        rho_left = np.repeat([Q_ref-.5*bin_size], outputs.shape[0], 0)
        rho_right = np.repeat([Q_ref+.5*bin_size], outputs.shape[0], 0)
        rho_left = np.all(np.greater_equal(outputs, rho_left), axis=1)
        rho_right = np.all(np.less_equal(outputs, rho_right),axis=1)
        inside = np.logical_and(rho_left, rho_right)
        max_values = np.repeat(maximum, outputs.shape[0], 0)
        return inside.astype('float64')*max_values


Given a (M, mdim) data vector
:class:`~bet.sampling.adaptiveSampling.rhoD_kernel` expects that ``rho_D``
will return a :class:`~numpy.ndarray` of shape (M,). 

Next we designate some refernce point in the parameter space::

    p_ref = mdat['points_true']
    p_ref = p_ref[5:7, 15]

We can visualize the samples in the parameter space::

    pDom.show_param(samples=points.transpose(), data=Q, rho_D=rho_D, p_ref=p_ref)

or the data space::

    pDom.show_data(data=Q, rho_D=rho_D, Q_ref=Q_ref)

We can also visualize the data domain :math:`Q(\Lambda)` that corresponds with
the convex hull of samples in the parameter space::

    pDom.show_data_domain_2D(samples=points.transpose(), data=Q, Q_ref=Q_ref)

We can compare multiple data domains that result from using different QoI maps::

    pDom.show_data_domain_multi(samples=points.transpose(), data=mdat['Q'],
        Q_ref=mdat['Q_true'][15], Q_nums=[1,2,5], showdim='all')

