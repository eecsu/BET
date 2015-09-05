#! /usr/bin/env python
# -*- coding: utf-8 -*-
# import necessary modules
import numpy as np
import bet.sampling.adaptiveSampling as asam
import bet.sampling.gradientSampling as oasam
import bet.sampling.basicSampling as bsam
import bet.sampling.smoothIndicatorFunction as sif
import scipy.io as sio
from scipy.interpolate import griddata
from bet.Comm import *
import bet.postProcess.postTools as pT
from matplotlib import pyplot as plt
import bet.util as util
import math, os
import bet.calculateP.calculateP as calcP
import bet.calculateP.simpleFunP as sfun
import bet.calculateP.indicatorFunctions as ifun
#import polyadcirc.pyGriddata.file_management as fm

size = comm.Get_size()
rank = comm.Get_rank()

# REMEMBER lambda_emulate differes across processors!!!

# Set minima and maxima
lam_domain = np.array([[.07, .15], [.1, .2]])
param_min = lam_domain[:, 0]
param_max = lam_domain[:, 1]

# Select only the stations I care about this will lead to better sampling
station_nums = [0, 1] # 1, 2

# Create Transition Kernel
transition_set = asam.transition_set(.5, .5**5, 1.0)

# Read in Q_ref and Q to create the appropriate rho_D 
mdat = sio.loadmat('Q_2D')
Q = mdat['Q']
Q = Q[:, station_nums]
Q_ref = mdat['Q_true']
Q_ref = Q_ref[15, station_nums] # 16th/20

# Create experiment model
points = mdat['points']
#xx = (points[0,:]+points[1,:]).transpose()
#yy = (points[0,:]-points[1,:]).transpose()
#Q = np.column_stack((xx, yy))
#Q = np.column_stack((points[1,:].transpose(), points[0,:].transpose()))
#Q = np.column_stack((xx, xx))[:,1]
#Q = points.transpose()
#Q[:,0] = xy
#Q = points.transpose()[:, 1]
#Q = util.fix_dimensions_data(Q, 1)
#print Q.shape
#Q_ref = np.mean(Q, 0)

bin_ratio = 0.10
bin_size = (np.max(Q, 0)-np.min(Q, 0))*bin_ratio
# Read in points_ref and plot results
p_ref = mdat['points_true']
p_ref = p_ref[5:7, 15]

# Create experiment model
points = mdat['points']
def model(inputs):
    interp_values = np.empty((inputs.shape[0], Q.shape[1])) 
    for i in xrange(Q.shape[1]):
        interp_values[:, i] = griddata(points.transpose(), Q[:, i],
                inputs)
    return interp_values 

sur_domain = np.empty((len(Q_ref),2))
sur_domain[:, 0] = np.min(Q, 0)
sur_domain[:, 1] = np.max(Q, 0)

# gradient sampling parameters
#cluster_type = 'ffd'
#cluster_type = 'cfd'
cluster_type = 'rbf'
radius_ratio = .01*bin_ratio
nominal_ratio = 0.5

# Create rho_D
maximum = 1/np.product(bin_size)
hrect = ifun.hyperrectangle_size(Q_ref, bin_size*1.125+3*radius_ratio)
def rho_D(outputs):
    inside = hrect(outputs)
    max_values = np.repeat(maximum, outputs.shape[0], 0)
    return inside.astype('float64')*max_values

def smooth_Ind(outputs):
    fun = sif.smoothed_indicator_cws(Q_ref, bin_size-3*radius_ratio, sur_domain)
    return fun(outputs)

bound_hrect = ifun.boundary_hyperrectangle_size_ratio(Q_ref, bin_size, 0.25)
def bound_rho_D(outputs):
    inside = bound_hrect(outputs)
    return inside.astype('float64')

def smooth_W(outputs):
    fun = sif.smoothed_indicator_W_cws(Q_ref, bin_size, sur_domain)
    return fun(outputs)


print "Sampling Parameter Space"
max_samples = 1e4 #5
# Create adaptive sampler
num_chains = 80
chain_length = int(math.ceil(max_samples/num_chains))
num_samples = chain_length*num_chains
idx = np.reshape(np.arange(num_samples), (num_chains, chain_length)).ravel('F')
adSampler = oasam.sampler(num_samples, chain_length, model)

inital_sample_type = "random"

# Create basic sampler
baSampler = bsam.sampler(model, num_samples)

# Post-process and compare volumes
num_l_emulate = max_samples * 100#0

# Create Simple function approximation
# Save points used to partion D for simple function approximation and the
# approximation itself (this can be used to make close cmparisions...)
(rho_D_M, d_distr_samples, d_Tree) = sfun.uniform_hyperrectangle(Q, Q_ref,
        bin_ratio=bin_ratio, center_pts_per_edge=np.ones((Q.shape[1],)))

def estimate_volume_set(set_number):
    folder = "set"+str(set_number)
    if rank == 0:
        os.mkdir(folder)
    comm.Barrier()
    lambda_emulate = calcP.emulate_iid_lebesgue(lam_domain, num_l_emulate)
    if rank == 0:
        print "Finished emulating lambda samples"

    # Get adaptive samples
    (gcsamples, gcdata, gcall_step_ratios) = adSampler.generalized_chains(param_min, 
            param_max, transition_set, rho_D, smooth_Ind, "adaptive_samples",
            radius_ratio=radius_ratio, cluster_type=cluster_type,
            nominal_ratio=nominal_ratio)#,
            #inital_sample_type=inital_sample_type)

    # Get boundary focused adaptive samples
    (bgcsamples, bgcdata, bgcall_step_ratios) = adSampler.generalized_chains(param_min, 
            param_max, transition_set, bound_rho_D, smooth_W,
            "boundary_samples", radius_ratio=radius_ratio,
            cluster_type=cluster_type, nominal_ratio=nominal_ratio)#, 
            #inital_sample_type=inital_sample_type)
    quit()

    # Get uniform random samples
    (usamples, udata) = baSampler.random_samples('random', param_min,
        param_max, "random_samples", parallel=True)

    # Estimate "true" volume using prob_emulated 
    _, o_data = baSampler.user_samples(lambda_emulate, "over_sampling")
    (P, _, _, _) = calcP.prob_emulated(lambda_emulate, o_data, rho_D_M,
        d_distr_samples, None, d_Tree)

    lam_vol4 = np.ones((lambda_emulate.shape[0],))*(1.0/float(num_l_emulate))
    vol_est4 = np.sum(lam_vol4[P.nonzero()])
    vol_est4 = comm.allreduce(vol_est4, vol_est4)

    if rank == 0:
        print "Estimated truth volume ", vol_est4 

    def estimate_target_volumes_adaptive_uniform(num_samples):
        """
        Calculate P on the actual samples estimating voronoi cell volume with MC
        integration
        """
        filename = os.path.join(folder, "estimateVoutput_nS"+str(num_samples))

        sample_idx = idx[:num_samples] 

        # ADAPTIVE SAMPLES
        (P1, lam_vol1, _, io_ptr1, emulate_ptr1) = calcP.prob_mc(gcsamples[sample_idx,:],
                gcdata[sample_idx,:], rho_D_M, d_distr_samples, 
                lambda_emulate, d_Tree)

        # BOUNDARY ADAPTIVE SAMPLES
        (P11, lam_vol11, _, io_ptr11, emulate_ptr11) = calcP.prob_mc(bgcsamples[sample_idx,:],
                bgcdata[sample_idx,:], rho_D_M, d_distr_samples, 
                lambda_emulate, d_Tree)

        # UNIFORM SAMPLES VOLUME EMULATION
        (P2, lam_vol2, _, io_ptr2, emulate_ptr2) = calcP.prob_mc(usamples[:num_samples,:], 
                udata[:num_samples,:], rho_D_M, d_distr_samples, 
                lambda_emulate, d_Tree)

        # UNIFORM SAMPLES
        (P22, lam_vol22, io_ptr22) = calcP.prob(usamples[:num_samples,:], 
                udata[:num_samples,:], rho_D_M, d_distr_samples, 
                d_Tree)

        # ESTIMATE VOLUMES
        vol_est1 = np.sum(lam_vol1[P1.nonzero()])
        vol_est11 = np.sum(lam_vol11[P11.nonzero()])
        vol_est2 = np.sum(lam_vol2[P2.nonzero()])
        #lam_vol22 = lam_vol22/num_samples
        vol_est22 = np.sum(lam_vol22[P22.nonzero()])

        if rank == 0:
            mdict = dict()
            mdict['vol_est1'] = vol_est1
            mdict['vol_est11'] = vol_est11
            mdict['vol_est2'] = vol_est2
            mdict['vol_est22'] = vol_est22
            mdict['P1'] = P1
            mdict['P11'] = P11
            mdict['lam_vol1'] = lam_vol1
            mdict['lam_vol11'] = lam_vol11
            mdict['P2'] = P2
            mdict['P22'] = P22
            mdict['lam_vol2'] = lam_vol2
            mdict['lam_vol22'] = lam_vol22
            # SAVE THING
            print "Finished estimating volume for {} samples".format(num_samples)
            sio.savemat(filename, mdict, do_compression=True)
            print num_samples, vol_est1, vol_est11, vol_est2, vol_est22
        return vol_est1, vol_est11, vol_est2, vol_est22



    nums = []
    vols1 = []
    vols11 = []
    vols2 = []
    vols22 = []

    for n in xrange(num_chains,int(num_samples),num_chains*5):
        if rank == 0:
            print "Starting estimating volume for {} samples".format(n)
        v1, v11, v2, v22 = estimate_target_volumes_adaptive_uniform(n)
        nums.append(n)
        vols1.append(v1)
        vols11.append(v11)
        vols2.append(v2)
        vols22.append(v22)

    nums = np.array(nums)
    vols1 = np.array(vols1)
    vols11 = np.array(vols11)
    vols2 = np.array(vols2)
    vols22 = np.array(vols22)

    vols4 = vol_est4 * np.ones(nums.shape)


    if rank == 0:
        mdict = dict()
        mdict['nums'] = nums
        mdict['vols1'] = vols1
        mdict['vols11'] = vols11
        mdict['vols2'] = vols2
        mdict['vols22'] = vols22
        mdict['vols4'] = vols4
        # SAVE THE THING
        sio.savemat(os.path.join(folder, "volume_estimates_wrgt"), mdict, do_compression=True)



estimate_volume_set(1)


