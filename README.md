# How to run this program on the dev server?
# step1. scp /local_path/main.cpp usrname@mcs1.wlu.ca:/home/usrname/dir
# step2. mpicxx -fopenmp -O2 /home/usrname/dir/main.cpp -o /home/usrname/dir/test.out
# step3. OMP_NUM_THREADS=3 mpirun -np 9 /home/usrname/dir/test.out
