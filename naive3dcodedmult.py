#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD # the default communicator which consists of all the processors
rank = comm.Get_rank() # Returns the process ID of the current process
size = comm.Get_size() # Returns the number of processes

fault_tolerance = 9
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

if rank == size-1:               # This is the master's task
    print("hola my name is ", rank)
    a = np.array([[1,2],[3,4]]) #This is the A matrix
    b = np.array([[5,6],[7,8]]) #This is the B matrix
    print("the matrix A is")
    print(a)
    print("the matrix B is")
    print(b)
    blockmatrix =np.empty([9,2],dtype=float) #Here the master is encoding the tasks
    for i in range(9):                       #that he is going to give to the workers
        blockmatrix[i] = np.array([acode(a,i),bcode(b,i)])
    print(blockmatrix)
    for i,submatrix in enumerate(blockmatrix):
         req = comm.Isend(submatrix, dest=i, tag=0)
         req.Wait()                  #The master now broadcasts the tasks to the workers

    results = np.empty([fault_tolerance],dtype=float)
    place_to_rank = []                      #The worker now recieves the tasks in whatever
    for i in range(fault_tolerance):        #order the workers finish in
        req = comm.irecv(source=MPI.ANY_SOURCE, tag=1)  #That is the purpose of MPI.ANY_SOURCE
        data = req.wait()
        print(data)
        print(data["result"],data["rank"])
        results[i] = data["result"]
        place_to_rank.append(int(data["rank"]))
    decoder = np.fromfunction(np.vectorize(lambda i ,j :  place_to_rank[i]**j), [fault_tolerance,fault_tolerance],dtype=int)
    # print(decoder)
    # print(place_to_rank)
    # print(results)
    finalresult = np.matmul(np.linalg.inv(decoder), results) #The decoder matrix represents the linear system of equations, i.e. (decoder)*(results)=finalsolution,
    print(np.rint(finalresult))                              #that the master must solve to decode the data. The master decodes the data by performing
    c = np.array([[finalresult[1],finalresult[5]],[finalresult[3],finalresult[7]]])#results=(decoder)^{-1}*(finalsolution) which is equivalent to solving the system of
    print(np.rint(c))                                        #equations. the indices 1,3,5,7 contain all of the desired data, the rest is trash

else:
    submatrix = np.array([1,2],dtype=float)           #The worker recieves the task from the master and
    req = comm.Irecv(submatrix ,source=size-1, tag=0) #performs the multiplication assigned to him
    req.Wait()
    result = submatrix[0]*submatrix[1]
    print("hola my name is ", rank,"my task is ", submatrix, "and  my result is ", result )
    data = {'result': result, 'rank': rank}
    req = comm.isend(data, dest=size-1, tag=1)        #The worker then sends the results back
    req.wait()                                        #with his rank so the master can identify him
