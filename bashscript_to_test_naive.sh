#!/bin/bash

max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 10 python3 silent_naive_square_3d_coded_mult.py 2 2 2 >> output_for_naive10.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 14 python3 silent_naive_square_3d_coded_mult.py 3 2 2 >> output_for_naive14.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 15 python3 silent_naive_square_3d_coded_mult.py 2 3 2 >> output_for_naive15.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 18 python3 silent_naive_square_3d_coded_mult.py 4 2 2 >> output_for_naive18.txt
done


max=5
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 20 python3 silent_naive_square_3d_coded_mult.py 2 4 2 >> output_for_naive20.txt
done

max=6
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 20 python3 silent_naive_square_3d_coded_mult.py 3 2 3 >> output_for_naive20.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 26 python3 silent_naive_square_3d_coded_mult.py 4 2 3 >> output_for_naive26.txt
done

max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 27 python3 silent_naive_square_3d_coded_mult.py 4 3 2 >> output_for_naive27.txt
done

max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 30 python3 silent_naive_square_3d_coded_mult.py 3 3 3 >> output_for_naive30.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 34 python3 silent_naive_square_3d_coded_mult.py 4 2 4 >> output_for_naive34.txt
done
