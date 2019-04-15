#!/usr/bin/env python

from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD # the default communicator which consists of all the processors
rank = comm.Get_rank() # Returns the process ID of the current process
size = comm.Get_size() # Returns the number of processes

matrix_size_parameter = 2**4

worker_size_parameter = 2

Left_matrix_real_size_dimension_one = matrix_size_parameter
Left_matrix_real_size_dimension_two = matrix_size_parameter
Right_matrix_real_size_dimension_one = matrix_size_parameter
Right_matrix_real_size_dimension_two = matrix_size_parameter

m=worker_size_parameter
p=worker_size_parameter
n=worker_size_parameter

fault_tolerance = m*p*n+p-1

x=Left_matrix_real_size_dimension_one//m
y=Left_matrix_real_size_dimension_two//p
z=Right_matrix_real_size_dimension_two//n



def acode(a,i):           #This is the encoding function for the "A" matrix or
    z = 0                 #the left matrix that is going to be multiplied
    for j in range(p):
        for k in range(m):
            z = z + a[k][j]*i**(j+2*k)
    return z

def bcode(a,i):          #This is the encoding function for the "B" matrix or
    z = 0                #the right matrix that is going to be multiplied
    for j in range(p):
        for k in range(n):
            z = z + a[j][k]*i**(1-j+4*k)
    return z

def psuedo_rank(place_rank_translator, my_rank):
    for i,x in enumerate(place_rank_translator):
        if x == my_rank:
            return i



if rank == size-1:               # This is the master's task
    start_time = MPI.Wtime()
    a = np.arange(m*p*x*y).reshape(m,p,x,y)+1 #This is the A matrix
    b = np.arange(p*n*y*z).reshape(p,n,y,z)+m*p*x*y+1#This is the B matrix
    blockmatrix =np.empty([size-1,2,x,y],dtype=float) #Here the master is encoding the tasks
    for i in range(size-1):                       #that he is going to give to the workers
        blockmatrix[i] = np.array([acode(a,i),bcode(b,i)])
    for i,submatrix in enumerate(blockmatrix):
         req = comm.Isend(submatrix, dest=i, tag=0)
         req.Wait()                  #The master now broadcasts the tasks to the workers

    place_to_rank = []                      #The worker now recieves the tasks in whatever
    for i in range(fault_tolerance):        #order the workers finish in
        req = comm.irecv(source=MPI.ANY_SOURCE, tag=1)  #That is the purpose of MPI.ANY_SOURCE
        data = req.wait()
        place_to_rank.append(int(data))
        comm.send(i, dest=place_to_rank[i], tag=2)
    for i in range(fault_tolerance):        #order the workers finish in
        comm.send(place_to_rank, dest=place_to_rank[i], tag=3)
    for i in range(size - fault_tolerance - 1):
        req = comm.irecv(source=MPI.ANY_SOURCE, tag=1)  #That is the purpose of MPI.ANY_SOURCE
        data = req.wait()
        place_to_rank.append(int(data))
        comm.send(i+fault_tolerance, dest=place_to_rank[i+fault_tolerance], tag=2)
    finalresult = np.empty([fault_tolerance,x,z],dtype=float)
    for i in range(fault_tolerance):        #order the workers finish in
        finalresult[i] = comm.recv(source=place_to_rank[i], tag=3+3*fault_tolerance)
    c = np.empty([m,n,x,z],dtype=float)                           #that the master must solve to decode the data. The master decodes the data by performing
    for i in range(m):                                       #equations. the indices 1,3,5,7 contain all of the desired data, the rest is trash
        for j in range(n):
            c[i][j]=finalresult[p-1+i*p+j*p*m]
    finish_time = MPI.Wtime()
    total_time = finish_time - start_time
    print(total_time)
    print("yepa")
    print(np.rint(c))
    print("qepa")
    print(np.einsum('iksr,kjrt->ijst', np.arange(m*p*x*y).reshape(m,p,x,y)+1 , np.arange(p*n*y*z).reshape(p,n,y,z)+m*p*x*y+1))






else:
    submatrix = np.empty([2,x,y],dtype=float)           #The worker recieves the task from the master and
    req = comm.Irecv(submatrix ,source=size-1, tag=0) #performs the multiplication assigned to him
    req.Wait()
    result = np.matmul(submatrix[0],submatrix[1])
    req = comm.isend(rank, dest=size-1, tag=1)        #The worker then sends the results back
    req.wait()                                        #with his rank so the master can identify him
    my_place = comm.recv(source=size-1, tag=2)
    if my_place >= fault_tolerance:
        exit()
    place_to_rank = comm.recv(source=size-1, tag=3)
    my_row = np.fromfunction(lambda i  :  rank**i, (fault_tolerance,),dtype=float)
    acummalator = result
    for i in range(fault_tolerance):
        if i < my_place:
            req = comm.irecv(source=place_to_rank[i], tag=3+i)
            new_row = req.wait()
            req = comm.irecv(source=place_to_rank[i], tag=3+i+fault_tolerance)
            new_mat = req.wait()
            acummalator = acummalator - my_row[i]*new_mat
            my_row = my_row - my_row[i]*new_row
        if i == my_place:
            acummalator = acummalator/my_row[i]
            my_row = my_row/my_row[i]
            for j in range(fault_tolerance-my_place-1):
                req = comm.isend(my_row, dest=place_to_rank[my_place+j+1], tag=3+i)
                req.wait()
                req = comm.isend(acummalator, dest=place_to_rank[my_place+j+1], tag=3+i+fault_tolerance)
                req.wait()
    for k in range(fault_tolerance):
        i=fault_tolerance-k-1
        if i > my_place:
            req = comm.irecv(source=place_to_rank[i], tag=3+i+2*fault_tolerance)
            new_mat = req.wait()
            acummalator = acummalator - my_row[i]*new_mat
        if i == my_place :
            if i != fault_tolerance-1:
                acummalator = acummalator/my_row[i]
            for j in range(my_place):
                req = comm.isend(acummalator, dest=place_to_rank[j], tag=3+i+2*fault_tolerance)
                req.wait()
    req = comm.isend(acummalator, dest=size-1, tag=3+3*fault_tolerance)        #The worker then sends the results back
    req.wait()
