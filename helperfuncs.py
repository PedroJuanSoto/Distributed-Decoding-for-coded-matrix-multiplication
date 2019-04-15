#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD # the default communicator which consists of all the processors
rank = comm.Get_rank() # Returns the process ID of the current process
size = comm.Get_size() # Returns the number of processes

fault_tolerance = 9
fault_tolerance = 9
m=2
p=2
n=2

x=2
y=2
z=2
# print("numpyver",np.__version__)


def acode(a,i):           #This is the encoding function for the "A" matrix or
    z = 0                 #the left matrix that is going to be multiplied
    for j in range(2):
        for k in range(2):
            z = z + a[k][j]*i**(j+2*k)
    return z

def bcode(a,i):          #This is the encoding function for the "B" matrix or
    z = 0                #the right matrix that is going to be multiplied
    for j in range(2):
        for k in range(2):
            z = z + a[j][k]*i**(1-j+4*k)
    return z

def left_matrix_to_buffer(left_matrix):
    buffet = np.empty(x*y,dtype=float)
    for s in range(x):
        for t in range(y):
            buffet[s*y+t] = left_matrix[s][t]
    return buffet

def right_matrix_to_buffer(right_matrix):
    buffet = np.empty(x*y,dtype=float)
    for s in range(y):
        for t in range(z):
            buffet[s*z+t] = right_matrix[s][t]
    return buffet


def buffer_to_left_matrix(buffet):
    left_matrix = np.empty([x,y],dtype=float)
    for s in range(x):
        for t in range(y):
            left_matrix[s][t] = buffet[s*y+t]
    return left_matrix

def buffer_to_right_matrix(buffet):
    right_matrix = np.empty([y,z],dtype=float)
    for s in range(x):
        for t in range(y):
            right_matrix[s][t] = buffet[s*z+t]
    return right_matrix



a = np.array([[1,2],[3,4]]) #This is the A matrix
b = np.array([[5,6],[7,8]]) #This is the B matrix


print(left_matrix_to_buffer(a))
print(right_matrix_to_buffer(b))
print(buffer_to_left_matrix(left_matrix_to_buffer(a)))
print(buffer_to_right_matrix(left_matrix_to_buffer(b)))
print(np.concatenate((left_matrix_to_buffer(a),right_matrix_to_buffer(b)),axis=0))
print(np.concatenate((left_matrix_to_buffer(a),right_matrix_to_buffer(b)),axis=0)[:x*y])
print(np.concatenate((left_matrix_to_buffer(a),right_matrix_to_buffer(b)),axis=0)[x*y:])
print(buffer_to_left_matrix(np.concatenate((left_matrix_to_buffer(a),right_matrix_to_buffer(b)),axis=0)[:x*y]))
print(buffer_to_right_matrix(np.concatenate((left_matrix_to_buffer(a),right_matrix_to_buffer(b)),axis=0)[x*y:]))
