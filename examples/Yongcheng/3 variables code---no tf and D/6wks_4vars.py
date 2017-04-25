# -*- coding: utf-8 -*-
#! /usr/bin/env python

# Copyright (C) 2014-2016 The BET Development Team

import numpy as np
import bet.calculateP.simpleFunP as simpleFunP
import bet.calculateP.calculateP as calculateP
import bet.postProcess.plotP as plotP
import bet.postProcess.plotDomains as plotD
import bet.sample as samp
import bet.sampling.basicSampling as bsam
from myModel_4vars import my_model

sampler = bsam.sampler(my_model)
input_samples = samp.sample_set(4)
input_samples.set_domain(np.array([[7.79, 14.48],
                                   [50, 300],
                                   [0.73, 2.72],
                                   [1.0, 2.0]]))
                                   
randomSampling = True
if randomSampling is True:
    input_samples = sampler.random_sample_set('random',
                                              input_samples,
                                              num_samples=1E5)
else:
    input_samples = sampler.regular_sample_set(input_samples,
                              num_samples_per_dim=[50, 50, 50, 50])
    
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

print np.min(my_discretization._output_sample_set._values)
print np.max(my_discretization._output_sample_set._values)

#Q_ref =  my_model(param_ref)
Q_ref =58.39
#plotD.scatter_2D_multi(input_samples, ref_sample= param_ref, showdim = 'all',
                      #filename = 'linearMap_ParameterSamples',
                       #file_extension = '.eps')
#plotD.show_data_domain_2D(my_discretization, Q_ref = Q_ref, file_extension='.eps')


simpleFunP.uniform_partition_normal_distribution(
                        data_set=my_discretization, 
                        Q_ref=Q_ref, std = 2.44, M = 20)
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
ref_param = np.array([11.14, 109.58, 1.724, 1.5])
ref_params = np.array([[10.82, 140.37, 1.678, 1.5],
                       [9.69, 109.13, 1.273, 1.5],
                       [12.8, 105.05, 1.736, 1.5],
                       [11.25, 83.79, 2.209, 1.5]])

# calculate 2d marginal probs
(bins, marginals2D) = plotP.calculate_2D_marginal_probs(input_samples,
                                                        nbins = 50)

plotP.plot_2D_marginal_probs(marginals2D, bins, input_samples, 
                             lam_ref = ref_param,
                             filename = "Deflection_NotSmooth",
                             file_extension = ".jpg", plot_surface=False)

# smooth 2d marginals probs (optional)
marginals2D = plotP.smooth_marginals_2D(marginals2D, bins, 
                                        sigma=[1.0,10,0.25,0.2])

# plot 2d marginals probs
plotP.plot_2D_marginal_contours(marginals2D, bins, input_samples, 
                             lam_ref = ref_param,
                             lam_refs = ref_params,
                             contour_num = 10,
                             contour_font_size = 20,
                             filename = "Deflection",
                             file_extension = ".jpg")

# calculate 1d marginal probs
(bins, marginals1D) = plotP.calculate_1D_marginal_probs(input_samples,
                                                        nbins = 50)

plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, 
                             lam_ref = ref_param,
                             filename = "Deflection_NotSmooth",
                             file_extension = ".jpg")

# smooth 1d marginal probs (optional)
marginals1D = plotP.smooth_marginals_1D(marginals1D, bins,
                                        sigma=[0.5,40,0.5, 0.2])
# plot 1d marginal probs
plotP.plot_1D_marginal_probs(marginals1D, bins, input_samples, 
                             lam_ref = ref_param,
                             filename = "Deflection",
                             file_extension = ".jpg")


