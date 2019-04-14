#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD # the default communicator which consists of all the processors
rank = comm.Get_rank() # Returns the process ID of the current process
size = comm.Get_size() # Returns the number of processes

def acode(a,i):
    z = 0
    for j in range(2):
        for k in range(2):
            z = z + a[k][j]*i**(j+2*k)
    return z

def bcode(a,i):
    z = 0
    for j in range(2):
        for k in range(2):
            z = z + a[j][k]*i**(1-j+4*k)
    return z

if rank == size-1:
# Master work
    decoder = np.fromfunction(lambda i,j: i**j, [9,9],dtype=int)
    print(decoder)
    print("hola my name is ", rank)
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    print("the matrix A is")
    print(a)
    print("the matrix B is")
    print(b)
    blockmatrix =np.empty([9,2],dtype=int)
    for i in range(9):
        blockmatrix[i] = np.array([acode(a,i),bcode(b,i)])
    print(blockmatrix)
    for i,submatrix in enumerate(blockmatrix):
         req = comm.Isend(submatrix, dest=i, tag=i)
         req.Wait()
    result = 0
    result = comm.gather(result, root=size-1)
    print(result)
    finalresult = np.matmul(np.linalg.inv(decoder),(np.delete(np.asarray(result),9)))
    print(np.rint(finalresult))
    print(np.delete(np.asarray(result),9))
    c = np.array([[finalresult[1],finalresult[5]],[finalresult[3],finalresult[7]]])
    print(np.rint(c))

else:
    print("hola my name is ", rank)
    submatrix = np.array([1,2],dtype=int)
    req = comm.Irecv(submatrix ,source=size-1, tag=rank)
    req.Wait()
    print(submatrix)
    result = submatrix[0]*submatrix[1]
    print("hola my name is ", result )
    result = comm.gather(result, root=size-1)
