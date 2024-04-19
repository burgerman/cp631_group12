### How to run this program on the dev server?
##### step1. Upload the following files to the dev server.
##### scp /local_path/main.cpp usrname@mcs1.wlu.ca:/home/usrname/dir
##### scp /local_path/dir_init.py usrname@mcs1.wlu.ca:/home/usrname/dir
##### scp /local_path/visualize.py usrname@mcs1.wlu.ca:/home/usrname/dir
##### Install the core lib of visualization: pip install bokeh
##### Make sure to execute dir_init.py in the same location as the main.cpp before the execution of the main program
##### step2. mpicxx -fopenmp -O2 /home/usrname/dir/main.cpp -o /home/username/dir/test.out
##### step3. OMP_NUM_THREADS=3 mpirun -np 9 /home/usrname/dir/test.out


### Steps to visualize the Forest.
##### Login to the course server and make sure you have bokeh and numoy libraries with the required versions(check requiremetns.txt file for information)
##### Run python script "bokeh serve --show visualize.py"
##### To Visualize the forest on the web browser you need to open your local machine's terminal and setup and ssh tunnel.
##### Tunnel cmd: "ssh -L 5006:localhost:5006 username@mcs1.wlu.ca"
##### Open your local web browser and hit this url: "http://localhost:5006/visb"
##### You will be able to visualise all the iterations of the Forest.
