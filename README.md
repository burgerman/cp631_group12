### How to run this program on the dev server?
### step1. scp /local_path/main.cpp usrname@mcs1.wlu.ca:/home/usrname/dir
### Make sure you create output and data directories in the same location before executing the code
### step2. mpicxx -fopenmp -O2 /home/usrname/dir/main.cpp -o /home/usrname/dir/test.out
### step3. OMP_NUM_THREADS=3 mpirun -np 9 /home/usrname/dir/test.out


### Steps to visualize the Forest.
### Login to the course server and make sure you have bokeh and numoy libraries with the required versions(check requiremetns.txt file for information)
### Run python script "bokeh serve --show visualize.py"
### To Visualize the forest on the web browser you need to open your local machine's terminal and setup and ssh tunnel.
### Tunnel cmd: "ssh -L 5006:localhost:5006 username@mcs1.wlu.ca"
### Open your local web browser and hit this url: "http://localhost:5006/visb"
### You will be able to visualise all the iterations of the Forest.
