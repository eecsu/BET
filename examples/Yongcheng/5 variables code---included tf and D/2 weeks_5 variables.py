#! /usr/bin/env python

# Copyright (C) 2014-2016 The BET Development Team

import numpy as np
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.sample as samp
import bet.sampling.basicSampling as bsam
from myModel_yongcheng import my_model

sampler = bsam.sampler(my_model)
input_samples = samp.sample_set(5)
input_samples.set_domain(np.array([[14.2, 18],
                                   [129.9, 260.4],
                                   [0.61, 2.07],
                                   [0.006, 0.007],
                                   [3.99, 4.08]]))
                                   
randomSampling = True
if randomSampling is True:
    input_samples = sampler.random_sample_set('random',
                                              input_samples,
                                              num_samples=1E5)
else:
    input_samples = sampler.regular_sample_set(input_samples,
                              num_samples_per_dim=[100, 300, 200])
    
MC_assumption = True
# Estimate volumes of Voronoi cells associated with the parameter samples
if MC_assumption is False:
    input_samples.estimate_volume(n_mc_points=1E5)
else:
    input_samples.estimate_volume_mc()

# Create the discretization object using the input samples
my_discretization = sampler.compute_QoI_and_create_discretization(input_samples,
                                               savefile = 'Deflection_discretization.txt.gz')

#param_ref = np.array([[20, 227, 16.7]])

#Q_ref =  my_model(param_ref)
Q_ref =68.32
#plotD.scatter_2D_multi(input_samples, ref_sample= param_ref, showdim = 'all',
                      #filename = 'linearMap_ParameterSamples',
                       #file_extension = '.eps')
#plotD.show_data_domain_2D(my_discretization, Q_ref = Q_ref, file_extension='.eps')


simpleFunP.uniform_partition_normal_distribution(
                        data_set=my_discretization, 
                        Q_ref=Q_ref, std = 2.4, M = 10)
#        cells_per_dimension = 3))
#randomDataDiscretization = False
#if randomDataDiscretization is False:
#    simpleFunP.regular_partition_uniform_distribution_rectangle_scaled(
#        data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.1,
#        cells_per_dimension = 3)
#else:
#    simpleFunP.uniform_partition_uniform_distribution_rectangle_scaled(
#        data_set=my_discretization, Q_ref=Q_ref, rect_scale=0.1,
#        M=50, num_d_emulate=1E5)

# calculate probablities
calculateP.prob(my_discretization)


########################################
# Post-process the results
########################################
ref_param = np.array([16.1, 195.14, 1.34, 0.0065, 4.031])
# calculate 2d marginal probs
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples,
                                                        nbins = 50)

# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, 
                                        sigma=[2,40,0.5,0.2,0.2])

# plot 2d marginals probs
plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, 
                             lam_ref = ref_param,
                             filename = "Deflection",
                             file_extension = ".jpg", plot_surface=False)

# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
                                                        nbins = 50)
# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins,
                                        sigma=[2,40,0.5,0.2,0.2])
# plot 2d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, 
                             lam_ref = ref_param,
                             filename = "Deflection",
                             file_extension = ".jpg")
