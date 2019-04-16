#!/usr/bin/env python

from mpi4py import MPI
import numpy as np
import sys


comm = MPI.COMM_WORLD # the default communicator which consists of all the processors
rank = comm.Get_rank() # Returns the process ID of the current process
size = comm.Get_size() # Returns the number of processes

matrix_size_parameter = 2**6

worker_size_parameter = 2

m= int(sys.argv[1])
p= int(sys.argv[2])
n= int(sys.argv[3])

Left_matrix_real_size_dimension_one = (matrix_size_parameter//2)*m
Left_matrix_real_size_dimension_two = (matrix_size_parameter//2)*p
Right_matrix_real_size_dimension_one = (matrix_size_parameter//2)*p
Right_matrix_real_size_dimension_two = (matrix_size_parameter//2)*n


fault_tolerance = m*p*n+p-1

x=Left_matrix_real_size_dimension_one//m
y=Left_matrix_real_size_dimension_two//p
z=Right_matrix_real_size_dimension_two//n


def acode(a,i):           #This is the encoding function for the "A" matrix or
    z = 0                 #the left matrix that is going to be multiplied
    for j in range(p):
        for k in range(m):
            z = z + a[k][j]*i**(j+p*k)
    return z

def bcode(a,i):          #This is the encoding function for the "B" matrix or
    z = 0                #the right matrix that is going to be multiplied
    for j in range(p):
        for k in range(n):
            z = z + a[j][k]*i**(p-1-j+k*p*m)
    return z


if rank == size-1:               # This is the master's task
    start_time = MPI.Wtime()
    a = np.arange(m*p*x*y).reshape(m,p,x,y)+1 #This is the A matrix
    b = np.arange(p*n*y*z).reshape(p,n,y,z)+m*p*x*y+1#This is the B matrix
    blockmatrix =np.empty([size-1,2,x,y],dtype=np.double) #Here the master is encoding the tasks
    for i in range(size-1):                       #that he is going to give to the workers
        blockmatrix[i] = np.array([acode(a,i),bcode(b,i)])
    for i,submatrix in enumerate(blockmatrix):
         req = comm.Isend(submatrix, dest=i, tag=0)
         req.Wait()                  #The master now broadcasts the tasks to the workers

    results = np.empty([fault_tolerance,x,z],dtype=np.double)
    place_to_rank = []                      #The worker now recieves the tasks in whatever
    for i in range(fault_tolerance):        #order the workers finish in
        req = comm.irecv(source=MPI.ANY_SOURCE, tag=1)  #That is the purpose of MPI.ANY_SOURCE
        data = req.wait()
        results[i] = data["result"]
        place_to_rank.append(int(data["rank"]))
    decoder = np.fromfunction(np.vectorize(lambda i ,j :  place_to_rank[int(i)]**j), (fault_tolerance,fault_tolerance),dtype=np.double)

    finalresult = np.empty([fault_tolerance,x,z],dtype=np.double)
    finalresult = np.einsum('ik,k...->i...', np.linalg.inv(decoder), results)


    c = np.empty([m,n,x,z],dtype=np.double)
    for i in range(m):
        for j in range(n):
            c[i][j]=finalresult[p-1+i*p+j*p*m]
    finish_time = MPI.Wtime()
    total_time = finish_time - start_time
    print(total_time)
    # print("yepa")
    # print(np.rint(c))
    # print("qepa")
    # print(np.einsum('iksr,kjrt->ijst', np.arange(m*p*x*y).reshape(m,p,x,y)+1 , np.arange(p*n*y*z).reshape(p,n,y,z)+m*p*x*y+1))
    # print("yuppa")
    # print(np.rint(finalresult))



else:
    submatrix = np.empty([2,x,y],dtype=np.double)           #The worker recieves the task from the master and
    req = comm.Irecv(submatrix ,source=size-1, tag=0) #performs the multiplication assigned to him
    req.Wait()
    result = np.matmul(submatrix[0],submatrix[1])
    data = {'result': result, 'rank': rank}
    req = comm.isend(data, dest=size-1, tag=1)        #The worker then sends the results back
    req.wait()                                        #with his rank so the master can identify him
