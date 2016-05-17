#! /usr/bin/env python

# Copyright (C) 2014-2015 The BET Development Team

"""
This example solves a stochastic inverse problem for a 
linear 3-to-2 map. We refer to the map as the QoI map, 
or just a QoI. We refer to the range of the QoI map as 
the data space. 

The 3-D input space is discretized with i.i.d. uniform
random samples. We refer to the input space as the 
parameter space, and use parameter to refer to a particular 
point (e.g., a particular random sample) in this space.

A reference parameter is used to define a reference QoI datum
and a uniform probability measure is defined on a small box
centered at this datum.

The measure on the data space is discretized either randomly
or deterministically, and this discretized measure is then 
inverted by BET to determine a probability measure on the 
parameter space whose support contains the measurable sets 
of probable parameters. 

We use emulation to estimate the measures of sets defined by
the random discretizations.

1D and 2D marginals are calculated, smoothed, and plotted.
"""

import numpy as np
import bet.calculateP as calculateP
import bet.postProcess as postProcess
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.sample as samp
import bet.sampling.basicSampling as bsam
from myModel import my_model

# Initialize 3-dimensional input parameter sample set object
input_samples = samp.sample_set(3)

# Set parameter domain
input_samples.set_domain(np.repeat([[0.0, 1.0]], 3, axis=0))

# Define the sampler that will be used to create the discretization 
# object, which is the fundamental object used by BET to compute
# solutions to the stochastic inverse problem
'''
Suggested changes for user:   
        
Try num_samples = 1E3 and 1E4. What happens when num_samples = 1E2?

Try using 'lhs' instead of 'random' in the sampler.
'''
sampler = bsam.sampler(my_model, num_samples=1E4)

# Generate samples on the parameter space
input_samples = sampler.random_sample_set('random', input_samples)
# Estimate volumes of Voronoi cells associated with the parameter samples
'''
Suggested changes for user:

Try different numbers of points to estimate the volume, and also
try comparing to the standard Monte Carlo assumption that all the
Voronoi cells have the same measure.
'''
MC_assumption = False
if MC_assumption is False:
    input_samples.estimate_volume_mc(n_mc_points=1E5)
else:
    input_samples.estimate_volume_uniform()

# Now create the discretization object using the input samples
###############################################################
'''
TO LINDLEY:: WE NEED TO HAVE THE SAMPLER ALSO JUST EVALUATE THE QOI AND RETURN AN
OUTPUT DATA SET. THE SIMPLEFUNP REQUIRES THE OUTPUT DATA SET NOT THE DISCRETIZATION

OR

TO STEVE:: WE SHOULD HAVE SIMPLEFUNP TAKE IN A DISCRETIZATION OBJECT AND USE THE
OUTPUT DATA SET WITHIN IT TO CREATE THE PROBABILITY DATA SET.
'''
##################################################################
my_discretization = sampler.create_random_discretization('random', input_samples,
                                               savefile = '3to2_discretization.txt.gz')

# Define the reference parameter
param_ref = np.array([0.5, 0.5, 0.5])
# Compute the reference QoI
Q_ref =  my_model(param_ref)

'''
Suggested changes for user:
    
Try different ways of discretizing the probability measure on D defined as a uniform
probability measure on a rectangle (since D is 2-dimensional).
    
unif_unif creates a uniform measure on a hyperbox with dimensions relative to the
size of the circumscribed hyperbox of the set D using the bin_ratio. A total of M samples
are drawn within a slightly larger scaled hyperbox to discretize this measure defining
M total generalized contour events in Lambda. The reason a slightly larger scaled hyperbox
is used to draw the samples to discretize D is because otherwise every generalized contour
event will have non-zero probability which obviously defeats the purpose of "localizing"
the probability within a subset of D.
    
uniform_hyperrectangle uses the same measure defined in the same way as unif_unif, but the
difference is in the discretization which is on a regular grid defined by center_pts_per_edge.
If center_pts_per_edge = 1, then the contour event corresponding to the entire support of rho_D
is approximated as a single event. This is done by carefully placing a regular 3x3 grid (since D=2 in
this case) of points in D with the center point of the grid in the center of the support of
the measure and the other points placed outside of the rectangle defining the support to define
a total of 9 contour events with 8 of them having exactly zero probability.
'''
deterministic_discretize_D = True
if deterministic_discretize_D is True:
    probability_set = simpleFunP.uniform_hyperrectangle(
                                     data_set=my_discretization._output_sample_set,
                                     Q_ref=Q_ref, bin_ratio=0.2,
                                     center_pts_per_edge = 5)
else:
    probability_set = simpleFunP.unif_unif(data_set=my_discretization._output_sample_set,
                                          Q_ref=Q_ref, M=50, bin_ratio=0.2,
                                          num_d_emulate=1E5)

my_discretization._output_probability_set = probability_set

# calculate probablities
calculateP.prob(my_discretization)

# calculate 2d marginal probs
'''
Suggested changes for user:
    
At this point, the only thing that should change in the plotP.* inputs
should be either the nbins values or sigma (which influences the kernel
density estimation with smaller values implying a density estimate that
looks more like a histogram and larger values smoothing out the values
more).
    
There are ways to determine "optimal" smoothing parameters (e.g., see CV, GCV,
and other similar methods), but we have not incorporated these into the code
as lower-dimensional marginal plots have limited value in understanding the
structure of a high dimensional non-parametric probability measure.
'''
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples,
                                                        nbins = 30)
# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, sigma=0.1)

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, filename = "linearMap",
                             file_extension = ".eps", plot_surface=False)

# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
                                                        nbins = 40)
# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins, sigma=0.1)
# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, filename = "linearMap",
                             file_extension = ".eps")





