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


if rank == size-1:               # This is the master's task
    print("hola my name is ", rank)
    a = np.arange(m*p*x*y).reshape(m,p,x,y)+1 #This is the A matrix
    # a = np.array([[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1]]).reshape(m,p,x,y) #This is the A matrix
    b = np.arange(p*n*y*z).reshape(p,n,y,z)+m*p*x*y+1#This is the B matrix
    print("the matrix A is")
    print(a)
    print("the matrix B is")
    print(b)
    blockmatrix =np.empty([size-1,2,x,y],dtype=float) #Here the master is encoding the tasks
    for i in range(size-1):                       #that he is going to give to the workers
        blockmatrix[i] = np.array([acode(a,i),bcode(b,i)])
    print(blockmatrix)
    for i,submatrix in enumerate(blockmatrix):
         req = comm.Isend(submatrix, dest=i, tag=0)
         req.Wait()                  #The master now broadcasts the tasks to the workers

    results = np.empty([fault_tolerance,x,z],dtype=float)
    place_to_rank = []                      #The worker now recieves the tasks in whatever
    for i in range(fault_tolerance):        #order the workers finish in
        req = comm.irecv(source=MPI.ANY_SOURCE, tag=1)  #That is the purpose of MPI.ANY_SOURCE
        data = req.wait()
        print(data)
        print(data["result"],data["rank"])
        results[i] = data["result"]
        place_to_rank.append(int(data["rank"]))
    decoder = np.fromfunction(np.vectorize(lambda i ,j :  place_to_rank[i]**j), [fault_tolerance,fault_tolerance],dtype=int)

    finalresult = np.empty([fault_tolerance,x,z],dtype=float)
    finalresult = np.einsum('ik,k...->i...', np.linalg.inv(decoder), results)


    print(np.rint(finalresult))
    c = np.empty([2,2,2,2],dtype=float)                           #that the master must solve to decode the data. The master decodes the data by performing
    for i in range(2):                                       #equations. the indices 1,3,5,7 contain all of the desired data, the rest is trash
        for j in range(2):
            c[i][j]=finalresult[2-1+i*2+j*2*2]
    print("yepa",np.rint(c))
    print(np.einsum('iksr,kjrt->ijst', np.arange(m*p*x*y).reshape(m,p,x,y)+1 , np.arange(p*n*y*z).reshape(p,n,y,z)+m*p*x*y+1))



else:
    submatrix = np.empty([2,2,2],dtype=float)           #The worker recieves the task from the master and
    req = comm.Irecv(submatrix ,source=size-1, tag=0) #performs the multiplication assigned to him
    req.Wait()
    print("takaka", submatrix)
    result = np.matmul(submatrix[0],submatrix[1])
    print("hola my name is ", rank,"my task is multiply", submatrix[0], "with", submatrix[1], "and  my result is ", result )
    data = {'result': result, 'rank': rank}
    req = comm.isend(data, dest=size-1, tag=1)        #The worker then sends the results back
    req.wait()                                        #with his rank so the master can identify him
