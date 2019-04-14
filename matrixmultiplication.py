#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD # the default communicator which consists of all the processors
rank = comm.Get_rank() # Returns the process ID of the current process
size = comm.Get_size() # Returns the number of processes



if rank == 0:
# Master work
    print("hola my name is ", rank)
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    print(a)
    print(b)
    blockmatrix =np.empty([8,2],dtype=float)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                blockmatrix[i*4+j*2+k] = np.array([a[i][k],b[k][j]])
    print(blockmatrix)
    i=1
    for submatrix in blockmatrix:
         req = comm.Isend(submatrix, dest=i, tag=i)
         req.Wait()
         i=i+1
    result = 0
    result = comm.gather(result, root=0)
    print(result)
    finalresult = np.empty([2,2],dtype=float)
    for i in range(2):
        for j in range(2):
            finalresult[i][j] = result[2*i+j+1]+ result[2*i+j+2]
    print(finalresult)



else:
    print("hola my name is ", rank)
    submatrix = np.array([1,2],dtype=float)
    req = comm.Irecv(submatrix ,source=0, tag=rank)
    req.Wait()
    print(submatrix)
    result = submatrix[0]*submatrix[1]
    result = comm.gather(result, root=0)
