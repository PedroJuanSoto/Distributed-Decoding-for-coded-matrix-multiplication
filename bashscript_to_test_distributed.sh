#!/bin/bash



max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 10 python3 silent_distributed_square_3d_coded_mult.py 2 2 2 >> output_for_distributed10.txt
done

max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 11 python3 silent_distributed_square_3d_coded_mult.py 2 2 2 >> output_for_distributed11.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 12 python3 silent_distributed_square_3d_coded_mult.py 2 2 2 >> output_for_distributed12.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 13 python3 silent_distributed_square_3d_coded_mult.py 2 2 2 >> output_for_distributed13.txt
done



max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 14 python3 silent_distributed_square_3d_coded_mult.py 3 2 2 >> output_for_distributed14.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 15 python3 silent_distributed_square_3d_coded_mult.py 2 3 2 >> output_for_distributed15.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 18 python3 silent_distributed_square_3d_coded_mult.py 4 2 2 >> output_for_distributed18.txt
done


max=5
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 20 python3 silent_distributed_square_3d_coded_mult.py 2 4 2 >> output_for_distributed20.txt
done

max=6
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 20 python3 silent_distributed_square_3d_coded_mult.py 3 2 3 >> output_for_distributed20.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 26 python3 silent_distributed_square_3d_coded_mult.py 4 2 3 >> output_for_distributed26.txt
done

max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 27 python3 silent_distributed_square_3d_coded_mult.py 4 3 2 >> output_for_distributed27.txt
done

max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 30 python3 silent_distributed_square_3d_coded_mult.py 3 3 3 >> output_for_distributed30.txt
done


max=10
for (( i=2; i <= $max; ++i ))
do
    mpirun -np 34 python3 silent_distributed_square_3d_coded_mult.py 4 2 4 >> output_for_distributed34.txt
done
