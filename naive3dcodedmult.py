#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD # the default communicator which consists of all the processors
rank = comm.Get_rank() # Returns the process ID of the current process
size = comm.Get_size() # Returns the number of processes

fault_tolerance = 9
print("numpyver",np.__version__)
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
    print("hola my name is ", rank)
    a = np.array([[1,2],[3,4]])
    b = np.array([[5,6],[7,8]])
    print("the matrix A is")
    print(a)
    print("the matrix B is")
    print(b)
    blockmatrix =np.empty([9,2],dtype=float)
    for i in range(9):
        blockmatrix[i] = np.array([acode(a,i),bcode(b,i)])
    print(blockmatrix)
    for i,submatrix in enumerate(blockmatrix):
         req = comm.Isend(submatrix, dest=i, tag=0)
         req.Wait()

    results = np.empty([fault_tolerance],dtype=float)
    place_to_rank = []
    for i in range(fault_tolerance):
        req = comm.irecv(source=MPI.ANY_SOURCE, tag=1)
        data = req.wait()
        print(data)
        print(data["result"],data["rank"])
        results[i] = data["result"]
        place_to_rank.append(int(data["rank"]))
    decoder = np.fromfunction(np.vectorize(lambda i ,j :  place_to_rank[i]**j), [fault_tolerance,fault_tolerance],dtype=int)
    print(decoder)
    print(place_to_rank)
    print(results)
    finalresult = np.matmul(np.linalg.inv(decoder), results)
    print(np.rint(finalresult))
    c = np.array([[finalresult[1],finalresult[5]],[finalresult[3],finalresult[7]]])
    print(np.rint(c))

else:
    print("hola my name is ", rank)
    submatrix = np.array([1,2],dtype=float)
    req = comm.Irecv(submatrix ,source=size-1, tag=0)
    req.Wait()
    print(submatrix)
    result = submatrix[0]*submatrix[1]
    print("hola my result is ", result )
    data = {'result': result, 'rank': rank}
    req = comm.isend(data, dest=size-1, tag=1)
    req.wait()
