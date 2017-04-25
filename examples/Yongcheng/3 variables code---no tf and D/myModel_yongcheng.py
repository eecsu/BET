# Copyright (C) 2016 The BET Development Team

# -*- coding: utf-8 -*-
#import numpy as np

#x = np.linspace(0,10,100)
#y = np.linspace(0,8,100)
#z = np.linspace(2,8,100)
#t=2
# Define a model that is a linear QoI map
def my_model(parameter_samples):
    B = parameter_samples[:,0]
    C = parameter_samples[:,1]
    F = parameter_samples[:,2]
    T = 0.0065
    D = 4.03
#    T = np.random.normal(0.0065, 0.00033, 6000000)
#    D = np.random.normal(4, 0.00033, 6000000)
    A = B+10*6.6*C*T*F/D #elementwise operations
    return A
