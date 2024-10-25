Collection of compression models used by group 5 for the course 2AMM20.

### MINER
Multiscale Implicit NEural Representation model (adapted from [here](https://github.com/vishwa91/MINER))

### NIF
Neural Imaging Format, adapted from the [NIF paper](https://doi.org/10.1145/3581783.3613834) and accompanying [Git Repository](https://github.com/aegroto/nif.git).

To run the code, the following steps are required:
* Download the 12 npy files from the dataset and store them in the directory `nif-master/experiments/images`
* Move the `my_job.sh` and `my_job3d.sh` files one directory up, such that it is contained in the same location as the `nif-master` directory.
* Run the `my_job.sh` file if you want to run the experiments on the images in their full 13-dimensional form. If you want to run the experiments on the rgb channels of the images only, run `my_job3d.sh`