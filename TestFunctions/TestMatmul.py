# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 06:39:35 2020

@author: VH Chen
"""

import numpy as np


def contract_special(arg_a,arg_b):
    # Takes a matrix of TxMxN and a vector of TxN  or TxM
    # For each T, contract the matrix with the vector
    # Or two vectors of size TxN
    # For each T, contract the indices N 
    # Covers the case that matmul and dot don't do very elegantly
    # Avoids having to use a for loop to iterate over T
    if (np.ndim(arg_a) == 3 and np.ndim(arg_b) == 2): # arg_a is the matrix and arg_b is the vector
        matrix = arg_a
        vector = arg_b
        intermediate_result = np.tensordot(matrix,vector, ((2), (1)))
        result = np.diagonal(intermediate_result, offset=0, axis1=0, axis2=2).transpose()
    elif (np.ndim(arg_a) == 2 and np.ndim(arg_b) == 3): # arg_a is the vector and arg_b is the matrix
        vector = arg_a
        matrix = arg_b
        intermediate_result = np.tensordot(matrix,vector, ((1), (1)))
        result = np.diagonal(intermediate_result, offset=0, axis1=0, axis2=2).transpose()
    elif (np.ndim(arg_a) == 2 and np.ndim(arg_b) == 2): # arg_a is the vector and arg_b is a vector
        vector1 = arg_a
        vector2 = arg_b
        intermediate_result = np.tensordot(vector1,vector2, ((1), (1)))
        result = np.diagonal(intermediate_result, offset=0, axis1=0, axis2=1).transpose()
    else: 
        print('Error: Invalid dimensions')
    return result




test_matrix = np.zeros([10,3,3])
test_matrix2 = np.zeros([10,3,3])
test_vector = np.zeros([10,3])
test_vector2 = np.zeros([10,3])

for ii in range(0,10):
    test_matrix[ii,0,0]  = 1
    test_matrix[ii,0,1]  = 2
    test_matrix[ii,0,2]  = 3
    test_matrix[ii,1,0]  = 4
    test_matrix[ii,1,1]  = 5
    test_matrix[ii,1,2]  = 6
    test_matrix[ii,2,0]  = 7
    test_matrix[ii,2,1]  = 8
    test_matrix[ii,2,2]  = 9    
    test_matrix2[ii,0,0]  = 3
    test_matrix2[ii,0,1]  = 1
    test_matrix2[ii,0,2]  = 2
    test_matrix2[ii,1,0]  = 5
    test_matrix2[ii,1,1]  = 7
    test_matrix2[ii,1,2]  = 6
    test_matrix2[ii,2,0]  = 4
    test_matrix2[ii,2,1]  = 9
    test_matrix2[ii,2,2]  = 8  
    test_vector[ii,0]    = 4
    test_vector[ii,1]    = 5
    test_vector[ii,2]    = 6
    test_vector2[ii,0]    = 1
    test_vector2[ii,1]    = 2
    test_vector2[ii,2]    = 3
       
test = matmul_special(test_vector,test_matrix)
print(test)

test2 = np.matmul(test_matrix,test_matrix2)
print(test2)

test3 = matmul_special(test_vector,test_vector2)
print(test3)
