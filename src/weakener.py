import numpy as np
import torch
import cvxpy

from collections import Counter

class Weakener(object):
    def __init__(self, true_classes):

        # Dimentions of the problem
        self.c = true_classes
        self.d = None
        self.h = None

        # Matrices
        self.M = None
        
        #For FB losses
        self.Mr = None
        self.Ml = None
        self.B = None
        self.B_opt = None
        self.F = None
        
        #For Backward losses
        self.Y = None
        self.Y_opt = None
        self.Y_conv = None
        self.Y_opt_conv = None

    def generate_M(self, model_class = 'pll', alpha = 1, beta = None, corr_p = 0.5, corr_n = None):
        self.corr_p = corr_p
        if corr_n == None:
            self.corr_n = corr_p
        else:
            self.corr_n = corr_n
        self.pll_p = corr_p
        
        if model_class == 'Noisy_Patrini_MNIST':
            # Noise is: 2 -> 7; 3 -> 8; 5 <-> 6; 7 -> 1
            #self.M = torch.tensor([[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
            M = np.array([[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 1. , 0. , 0. , 0. , 0. , 0. , self.corr_p, 0. , 0. ],
                      [0. , 0. , 1-self.corr_p, 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 1-self.corr_p, 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 1-self.corr_p, self.corr_p, 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , self.corr_p, 1-self.corr_p, 0. , 0. , 0. ],
                      [0. , 0. , self.corr_p, 0. , 0. , 0. , 0. , 1-self.corr_p, 0. , 0. ],
                      [0. , 0. , 0. , self.corr_p, 0. , 0. , 0. , 0. , 1. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ]])
            # Noise_l is: 2 -> 7; 7 -> 1
            self.Ml = np.array([[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 1. , 0. , 0. , 0. , 0. , 0. , self.corr_p, 0. , 0. ],
                      [0. , 0. , 1-self.corr_p, 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 1, 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 1, 0, 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0, 1, 0. , 0. , 0. ],
                      [0. , 0. , self.corr_p, 0. , 0. , 0. , 0. , 1-self.corr_p, 0. , 0. ],
                      [0. , 0. , 0. , 0, 0. , 0. , 0. , 0. , 1. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ]])
            # Noise_l is: 3 -> 8; 5 <-> 6
            self.Mr = np.array([[1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 1. , 0. , 0. , 0. , 0. , 0. , 0, 0. , 0. ],
                      [0. , 0. , 1, 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 1-self.corr_p, 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 1-self.corr_p, self.corr_p, 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , self.corr_p, 1-self.corr_p, 0. , 0. , 0. ],
                      [0. , 0. , 0, 0. , 0. , 0. , 0. , 1, 0. , 0. ],
                      [0. , 0. , 0. , self.corr_p, 0. , 0. , 0. , 0. , 1. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ]])
        elif model_class == 'Noisy_Patrini_CIFAR10':
            #TRUCK → AUTOMOBILE, BIRD → AIRPLANE, DEER → HORSE, CAT ↔ DOG.
            #self.M = torch.tensor([[1. , 0. , self.corr_p , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
            M = np.array([[1. , 0. , self.corr_p , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , self.corr_p ],
                      [0. , 0. , 1-self.corr_p , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 1-self.corr_p , 0. , self.corr_p , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 1-self.corr_p , 0. , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , self.corr_p , 0. , 1-self.corr_p , 0. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , self.corr_p , 0. , 0. , 1. , 0. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. ],
                      [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1-self.corr_p ]])
            # Noise_l TRUCK → AUTOMOBILE, BIRD → AIRPLANE, 
            self.Ml = np.array([
                    [1. , 0. , self.corr_p , 0. , 0. , 0. , 0. , 0. , 0. , 0. ], 
                    [0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , self.corr_p ],
                    [0. , 0. , 1-self.corr_p , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1-self.corr_p ]
                ])
            # Noise_l DEER → HORSE, CAT ↔ DOG
            self.Mr = np.array([
                    [1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ], 
                    [0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 1-self.corr_p , 0. , self.corr_p , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 1-self.corr_p , 0. , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , self.corr_p , 0. , 1-self.corr_p , 0. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , self.corr_p , 0. , 0. , 1. , 0. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. , 0. ],
                    [0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 1. ]
                ])
        elif model_class == 'Noisy_CIFAR100':
            M = np.array([
                    [1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,     0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,   1-self.corr_p,   0,      0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,   1-self.corr_p,   0,      0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,   1-self.corr_p,   0,      0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,   1-self.corr_p,   0,      0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,   1-self.corr_p,   0],
                    [0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,      0,    self.corr_p,   1-self.corr_p]])
            self.Ml = np.array([
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p]])
            self.Mr = np.array([
                [1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, self.corr_p, 1 - self.corr_p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
        elif model_class == 'Noisy_Natarajan':
            #self.M = torch.tensor([
            M = np.array([
                [1-self.corr_n, self.corr_p  ],
                [self.corr_n  , 1-self.corr_p]])
        #elif model_class == 'Noisy_Natarajan':
        elif model_class == 'Decomposable_noisy_binary_0.2_0.2':
            M = np.array([
                [0.8, 0.2],
                [0.2, 0.8]
                ])
            self.Ml = np.array([
                [0.9, 0.1],
                [0.1, 0.9]])
            self.Mr = np.array([
                [0.875, 0.125],
                [0.125, 0.875]])
        #elif model_class == 'Noisy_Natarajan':
        elif model_class == 'Decomposable_noisy_binary_0.3_0.1':
            M = np.array([
                [0.9, 0.3],
                [0.1, 0.7]
                ])
            self.Ml = np.array([
                [0.95, 0.15],
                [0.05, 0.85]])
            self.Mr = np.array([
                [0.9375, 0.1875],
                [0.0625, 0.8125]])
        #elif model_class == 'Noisy_Natarajan':
        elif model_class == 'Decomposable_noisy_binary_0.4_0.4':
            M = np.array([
                [0.6, 0.4],
                [0.4, 0.6]
                ])
            self.Ml = np.array([
                [0.9, 0.1],
                [0.1, 0.9]])
            self.Mr = np.array([
                [0.625, 0.375],
                [0.375, 0.625]])
        elif model_class == 'pu':
            if self.c > 2:
                raise NameError('PU corruption coud only be applied when tne number o true classes is 2')
                # [TBD] if alpha is a vector raise error
                alpha = [alpha, 0]
            M = np.eye(2) + alpha * np.ones((2, 2))
            M /= np.sum(M, 0)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M
        # c = d
        elif model_class == 'supervised':
            M = np.identity(self.c)
            #self.M, self.Z, self.labels = self.label_matrix(M)

        elif model_class == 'noisy':
            '''
            - alpha > -1
                the limit case alpha = -1: Complemetary labels
                if alpha < 0: The false classes are more probable
                if alpha = 0: all classes are equally probable
                if alpha > 0: The true class is more prbable. 
                As a limiting case supervised is achieved as alpha -> infty
            [  1+a_1  a_2    a_3  ]
            [  a_1    1+a_2  a_3  ]
            [  a_1    a_2    1+a_3]
            '''
            if any(np.array(alpha) < -1):
                NameError('For noisy labels all components of alpha should be greater than -1')
            elif any(np.array(alpha) == -1):
                cl = np.where(np.array(alpha) == -1)[0]
                print('labels', cl, 'are considered complemetary labels')
                # warning('Some (or all) of the components is considered as complemetary labels')
            M = np.eye(self.c) + alpha * np.ones(self.c)
            M /= np.sum(M, 0)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M

        elif model_class == 'unif_noise':
            M = np.eye(self.c)*(1-self.corr_p-self.corr_p/(self.c - 1)) + np.ones(self.c)*self.corr_p/(self.c - 1)
            M /= np.sum(M, 0)
            self.Ml = self.factorize_stochastic_general()
            self.Mr = self.factorize_stochastic_general()

        elif model_class == 'complementary':
            '''
            This gives one of de non correct label 
            '''
            M = (1 - np.eye(c)) / (c - 1)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M
            Z = np.array([[int(x) for x in list(bin(i)[2:].zfill(c))] 
              for i in range(2**c) if bin(i).count('1') == 2])
            self.Mr = (1/(c-1)) * Z
            self.Ml = (1/(c-2)) * (1 - Z).T
        # c < d
        elif model_class == 'weak':
            '''
            - alpha > -1
                the limit case alpha = -1: Complemetary labels
                if alpha < 0: The false classes are more probable
                if alpha = 0: all classes are equally probable
                if alpha > 0: The true class is more probable. 
                As a limiting case supervised is achieved as alpha -> infty

             z\y  001    010    100
            000[  a_1    a_2    a_3  ]
            001[  1+a_1  a_2    a_3  ]
            010[  a_1    1+a_2  a_3  ]
            001[  a_1    a_2    a_3  ]
            011[  a_1    a_2    a_3  ]
            100[  a_1    a_2    1+a_3]
            101[  a_1    a_2    a_3  ]
            111[  a_1    a_2    a_3  ]
            '''
            M = np.zeros((2 ** self.c, self.c))
            for i in range(self.c):
                M[2 ** i, i] = 1
            M = alpha * M + np.ones((2 ** self.c, self.c))
            M /= np.sum(M, 0)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            #self.M = M
        elif model_class == 'pll':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021)
            probs, Z = self.pll_weights(p=self.pll_p)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            M = M / M.sum(0)
            #self.M, self.Z, self.labels = self.label_matrix(M)

        elif model_class == 'pll_a':
            # Mixing matrix for making pll corruption similar to that in
            # Instance-Dependent PLL (Xu, et al. 2021) they don't allow anchor points but this method does.
            probs, Z = self.pll_weights(p=self.pll_p, anchor_points=True)  # Take this probability from argparse

            M = np.array([list(map(probs.get, Z[:, i] * np.sum(Z, 1))) for i in range(self.c)]).T
            M = M / M.sum(0)
            #self.M, self.Z, self.labels = self.label_matrix(M)

        elif model_class == 'partial':
                self.corr_p = int(self.corr_p)
                Z = np.array([[int(i) for i in format(j,'b').zfill(self.c)] for j in range(2**self.c)])
                TL = np.identity(self.c,dtype='int')[::-1]
                M = np.zeros((2 ** self.c, self.c))
                for j,t_lab in enumerate(TL):
                    for i, w_lab in enumerate(Z):
                        if np.all(t_lab == w_lab):
                            M[i,j] += self.corr_p
                        elif np.dot(t_lab, w_lab) >0:
                            M[i,j] += (1-self.corr_p)/(2**(self.c-1)-1)
            
    
                M = M/np.sum(M, axis=0,keepdims=True)

        elif model_class == 'Complementary':
            '''
            This gives a set of candidate labels over the non correct one.
            '''
            M =  np.ones(self.c) - np.eye(self.c)
            M = M / M.sum(0)
            Z = np.array([[int(x) for x in list(bin(i)[2:].zfill(self.c))] 
              for i in range(2**self.c) if bin(i).count('1') == 2])
            self.Mr = (1/(self.c-1)) * Z
            self.Ml = (1/(self.c-2)) * (1 - Z).T
        elif model_class == 'clothing1m':
            M = np.eye(self.c)
            #self.M, self.Z, self.labels = self.label_matrix(M)
            def load_labels(filepath):
                """Loads image path -> label mapping from a file."""
                labels = {}

                with open(filepath, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 2:

                            image_path = os.path.normpath(parts[0])
                            labels[image_path] = int(parts[1])
                        else:

                            pass # print(f"Warning: Skipping malformed line in {filepath}: {line.strip()}")
                return labels
            def load_key_list(filepath):
                """Loads a list of image paths from a file."""
                keys = []

                with open(filepath, 'r') as f:
                    for line in f:

                        image_path = os.path.normpath(line.strip())
                        if image_path: # Ensure line is not empty
                            keys.append(image_path)
                return keys
            
            def estimate_transition_matrix(clean_labels_path, noisy_labels_path, num_classes):
                """
                Estimates the transition matrix T where T[i, j] = P(noisy=j | clean=i)
                by comparing full clean and noisy label mappings.

                Args:
                    clean_labels_path (str): Path to the clean label file (e.g., clean_label_kv.txt).
                    noisy_labels_path (str): Path to the noisy label file (e.g., noisy_label_kv.txt).
                    num_classes (int): The total number of classes.

                Returns:
                    numpy.ndarray: The estimated transition matrix (num_classes x num_classes),
                                or None if loading fails.
                """
                print("Loading clean labels from file...")
                clean_labels = load_labels(clean_labels_path)
                print("Loading noisy labels from file...")
                noisy_labels = load_labels(noisy_labels_path)

                if clean_labels is None or noisy_labels is None:
                    print("Error: Failed to load one or both label files. Cannot estimate matrix.")
                    return None

                if num_classes <= 0:
                    print("Error: num_classes must be positive.")
                    return None

                print(f"Found {len(clean_labels)} clean label entries and {len(noisy_labels)} noisy label entries.")

                common_keys = set(clean_labels.keys()) & set(noisy_labels.keys())
                print(f"Found {len(common_keys)} images present in both clean and noisy label sets.")

                if not common_keys:
                    print("Error: No common images found between clean and noisy sets. Cannot estimate matrix.")
                    return None

                count_matrix = np.zeros((num_classes, num_classes), dtype=float)
                invalid_labels_count = 0
                for key in common_keys:
                    clean_label = clean_labels[key]
                    noisy_label = noisy_labels[key]
                    if 0 <= clean_label < num_classes and 0 <= noisy_label < num_classes:
                        count_matrix[noisy_label,clean_label] += 1.0
                    else:
                        invalid_labels_count += 1
                if invalid_labels_count > 0:
                    print(f"Warning: Skipped {invalid_labels_count} pairs due to invalid label indices (out of bounds).")

                print("\nRaw Count Matrix (Rows=Clean, Cols=Noisy):")
                with np.printoptions(threshold=np.inf, suppress=True):
                    print(count_matrix.astype(int))

                epsilon = 1e-8
                zero_sum_rows = np.where(row_sums < epsilon)[0]
                if len(zero_sum_rows) > 0:
                    print(f"\nWarning: The following clean classes had no samples in the clean/noisy intersection:")
                    print(f"Indices: {zero_sum_rows}")
                    if category_names: print(f"Names: {[category_names[i] for i in zero_sum_rows]}")
                    print("Their rows in the transition matrix will be zero.")

                transition_matrix = count_matrix / np.sum(count_matrix, axis=0, keepdims=True)
                return transition_matrix
            metadata_dir = '/export/usuarios_ml4ds/danibacaicoa/ForwardBackard_losses_old/Datasets/raw_datasets/Clothing1M/'
            clean_label_kv_path = os.path.join(metadata_dir, 'clean_label_kv.txt')
            noisy_label_kv_path = os.path.join(metadata_dir, 'noisy_label_kv.txt')
            M = estimate_transition_matrix(clean_label_kv_path, noisy_label_kv_path, 14)

            n = M.shape[0]
            eps = 1e-3
            lr = 1e-2

            # Initialize A in (eps, 1-eps), B >= 0, both column-stochastic
            rng = np.random.default_rng(42)
            A = rng.uniform(eps, 1-eps, size=(n,n))
            A /= A.sum(axis=0, keepdims=True)
            B = rng.random((n,n))
            B /= B.sum(axis=0, keepdims=True)

            # Alternating projected gradient descent
            err = 999
            it = 0
            while err > 3e-2:
                # gradient w.r.t. A
                gradA = (A.dot(B) - M).dot(B.T)
                A -= lr * gradA
                # project A to [eps,1-eps] and column-normalize
                A = np.clip(A, eps, 1-eps)
                A /= A.sum(axis=0, keepdims=True)

                # gradient w.r.t. B
                gradB = A.T.dot(A.dot(B) - M)
                B -= lr * gradB
                # project B to [0, inf) and column-normalize
                B = np.maximum(B, 0)
                B /= B.sum(axis=0, keepdims=True)

                err = np.linalg.norm(A.dot(B) - M, 'fro')
                it += 1
                if it % 100 == 0:
                    print(f"Iteration {it}: Frobenius error = {err:.6e}")
            # Final error
            error = np.linalg.norm(A.dot(B) - M, 'fro')
            self.Mr = A
            self.Ml = B


        self.M, self.Z, self.labels = self.label_matrix(M)
        self.d = self.M.shape[0]
        
    def generate_weak(self, y, seed=None):
        # It should work with torch
        # the version of np.random.choice changed in 1.7.0 that could raise an error-
        d, c = self.M.shape
        # [TBD] include seed
        if np.array(y).ndim == 1:
            self.z = torch.Tensor([np.random.choice(d, p=self.M[:, tl]) for tl in y]).to(torch.int32)
        else:
            self.z = torch.Tensor([np.random.choice(d, p=self.M[:, tl]) for tl in torch.max(y, axis=1)[1]]).to(torch.int32)

        self.w = torch.from_numpy(self.Z[self.z.to(torch.int32)] + 0.)
        self.Y = np.linalg.pinv(self.M)
        self.Y_opt = self.virtual_matrix(p=None, optimize = True, convex=False)
        self.Y_conv = self.virtual_matrix(p=None, optimize = False, convex=True)
        self.Y_opt_conv = self.virtual_matrix(p=None, optimize = True, convex=True)

        return self.z, self.w


    def virtual_matrix(self, p=None, optimize = True, convex=True):
        d, c = self.M.shape
        I_c = np.eye(c)

        if p == None:
            if optimize:
                p = self.generate_wl_priors(self.z)
            else:
                p = np.ones(d)/d
        c_1 = np.ones((c,1))
        d_1 = np.ones((d,1))

        hat_Y = cvxpy.Variable((c,d))

        if c==d:
            Y = np.linalg.pinv(self.M)
        elif convex:
            prob = cvxpy.Problem(cvxpy.Minimize(
                cvxpy.norm(cvxpy.hstack([cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]),1)
            ),
                [hat_Y @ self.M == I_c, hat_Y.T @ c_1 == d_1]
            )
            prob.solve(solver=cvxpy.CLARABEL) #For cifar 100 it is not working
            #problem.solve(solver=cvxpy.ECOS)
            Y = hat_Y.value
        else:
            prob = cvxpy.Problem(cvxpy.Minimize(
                cvxpy.norm(cvxpy.hstack([cvxpy.norm(hat_Y[:, i])**2 * p[i] for i in range(d)]),1)
            ),
                [hat_Y @ self.M == I_c]
            )
            prob.solve(solver=cvxpy.CLARABEL)
            #problem.solve(solver=cvxpy.ECOS)
            Y = hat_Y.value
        
        return Y



    def virtual_labels(self, y = None, p=None, optimize = True, convex=True):
        '''
        z must be the weak label in the z form given by generate weak
        '''
        #In order to not generate weak labels each time we seek the existence of them
        # and in the case they are already generated we don't generate them again
        if self.z is None:
            if y is None:
                raise NameError('The weak labels have not been yet created. You shuold give the true labels. Try:\n  class.virtual_labels(y)\n instead')
            _,_ = self.generate_weak(y)
        if self.Y is None:
            self.virtual_matrix(p, optimize, convex)
        self.v = self.Y.T[self.z]
        return
    def generate_wl_priors(self, loss = 'CELoss'):

        #z_count = Counter(z)
        #p_est = np.array([z_count[x] for x in range(self.d)])
        p_est = np.array(self.z.bincount(minlength=self.Z.shape[0]))
        #p_est = np.array(torch.bincount(self.z))
        v_eta = cvxpy.Variable(int(self.c))
        if loss == 'CELoss':
            lossf = -p_est @ cvxpy.log(self.M @ v_eta)
        else:
            p_est = p_est / np.sum(p_est)
            lossf = cvxpy.sum_squares(p_est - self.M @ v_eta)

        problem = cvxpy.Problem(cvxpy.Minimize(lossf),
                                [v_eta >= 0, np.ones(self.c) @ v_eta == 1])
        problem.solve(solver=cvxpy.CLARABEL)

        # Compute the wl prior estimate
        p_reg = self.M @ v_eta.value

        return p_reg
    '''
        v_eta = cvxpy.Variable(self.c)
        if loss == 'cross_entropy':
            lossf = -p_est @ cvxpy.log(self.M @ v_eta)
        elif loss == 'square_error':
            p_est = p_est / np.sum(p_est)
            lossf = cvxpy.sum_squares(p_est - self.M @ v_eta)
        '''

    def label_matrix(self, M):
        """
        The objective of this function is twofold:
            1. It removes rows with no positive elements from M
            2. It creates a label matrix and a label dictionary

        Args:
            M (numpy.ndarray): A mixing matrix (Its not required an stochastic matrix).
                but its required its shape to be either dxc(all weak labels) or cxc(all true labels)

        Returns:
            - numpy.ndarray: Trimmed verison of the mixing matrix.
            - numpy.ndarray: Label matrix, where each row is converted to a binary label.
            - dict: A dictionary of labels where keys are indices and values are binary labels.

        Example:
            >>> M = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0],
                              [0, 1, 1], [1, 0, 1], [0, 1, 0],
                              [0, 1, 1], [0, 0, 0]])
            >>> trimmed_M, label_M, labels = label_matrix(M)
            >>> trimmed_M
            array([[1 0 0],
                   [0 1 1],
                   [1 0 1],
                   [0 1 0],
                   [0 1 1]])
            >>> label_M
            array([[0 0 0],
                   [0 1 1]
                   [1 0 0]
                   [1 0 1]
                   [1 1 0]])
            >>> labels
            {0: '000', 1: '011', 2: '100', 3: '101', 4: '110'}
        """
        d, c = M.shape

        if d == c:
            # If M is a square matrix, labels are
            Z = np.eye(c)
            # We make this in reversin order to get ('10..00', '01..00', .., '00..01')
            labels = {i: format(2**(c-(i+1)),'b').zfill(c) for i in range(c)} 
        elif (d<2**c):
            raise ValueError("Labels cannot be assigned to each row")
        else:
            # Z is a matrix with all the possible labels
            Z = np.array([[int(i) for i in format(j,'b').zfill(c)] for j in range(2**c)])
            # Now, we will get only the rows with nonzero elements
            z_row = M.any(axis = 1)
            # We assing the binary representation to those nonzero rows
            encoding = [format(i,'b').zfill(c) for i, exists in enumerate(z_row) if exists]
            # and we will give a numerical value to those representation of labels
            labels = {i:enc for i,enc in enumerate(encoding)}
            Z = Z[z_row,:]
            M = M[z_row,:]

        return M, Z, labels

    def pll_weights(self, c=None , p=0.5, anchor_points=False):
        """
        Descrip

        Args:
            p (double): 

        Returns:
            - dict: 
            - numpy.ndarray: 

        Example:
            >>> p = 
            >>> probs, Z= label_matrix(pll_weights)
            >>> probs
            output
            >>> z
            out
        """
        if c is None:
            c = self.c
        _, Z, _ = self.label_matrix(np.ones((2 ** c, c)))
        probs = {0: 0}
        q = 1 - p
        
        if anchor_points:
            probs[1] = q ** c + p * q ** (c - 1)
            probs[2] = p ** 2 * q ** (c - 2) + p * q ** (c - 1)
        else:
            probs[1] = 0
            probs[2] = p ** 2 * q ** (c - 2) + p * q ** (c - 1) + (q ** c + p * q ** (c - 1)) / (c - 1)
        for i in range(1, c + 1):
            probs[i] = p ** i * q ** (c - i) + p ** (i - 1) * q ** (c - i + 1)
        return probs, np.array(Z)
    
    def factorize_stochastic_general(self):
        c = self.c
        corr_p = self.corr_p
        if c < 2:
            raise ValueError("c must be at least 2.")
        if not (0 <= corr_p <= (c-1)/c):
            raise ValueError(f"corr_p must be in [0, {(c-1)/c:.4f}] for a valid factorization.")
        M_direct = (1 - corr_p) * np.eye(c) + (corr_p/(c-1)) * np.ones((c, c))

        a = c*(c-1)
        b = -2*(c-1)
        cc = corr_p 

        discriminant = b*b - 4*a*cc  
        if discriminant < 0:
            # Just to guard against floating precision issues:
            discriminant = 0.0

        sqrt_disc = np.sqrt(discriminant)

        # Two possible roots:
        o1 = (-b + sqrt_disc) / (2*a)  # "plus" root
        o2 = (-b - sqrt_disc) / (2*a)  # "minus" root

        # We'll pick whichever of o1 or o2 yields 0 <= o <= 1/(c-1) and 
        # also 0 <= d = 1 - (c-1)*o <= 1
        def valid(o):
            d_test = 1 - (c-1)*o
            return (o >= 0) and (d_test >= 0) and (o <= 1) and (d_test <= 1)

        candidates = []
        for o_candidate in (o1, o2):
            if valid(o_candidate):
                candidates.append(o_candidate)

        if not candidates:
            raise ValueError(
                "No nonnegative solution for (d, o) found. This can happen if corr_p is too large."
            )
        
        o = min(candidates)
        d = 1 - (c-1)*o
        X = np.full((c, c), o, dtype=float)
        np.fill_diagonal(X, d)

        #M_computed = X @ X
        return X