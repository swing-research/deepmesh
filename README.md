# Deepmesh
Repository for 'Deep Mesh Projectors for Inverse Problems'

We intend to make it as simple as possible to reproduce our results. If something is missing or you would like some guidance, please reach out to us.

## Summary
- In this work, we come up with a learning scheme to regularize ill-posed inverse problems. Instead of learning to go from measurements to the model directly, we learn to estimate certain random projections of the model. Specifically, we estimate projections on many random Delaunay triangulations of the model. Later, we combine them using regularized iterative schemes.
- As an example of very ill-posed inverse problem we choose the traveltime tomography problem, where a few sensors are placed on the image domain and line-integrals are calculated along lines connecting pairs of sensors. 
- We want to be able to work in environments where data is extremely scarce. Hence, we train on standard datasets ([LSUN](https://arxiv.org/abs/1506.03365), [celebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) etc.) and test on arbitrary image patches from [BP2004](http://software.seg.org/datasets/2D/2004_BP_Vel_Benchmark/) velocity dataset. We show that our method does not try to *paint* dataset-specific structures, instead it captures the global shapes much more faithfully. We use the signal-to-noise ratio as a metric to quantify the quality of our reconstructions.

### Scheme
We use neural network architectures inspired by U-nets. All networks are given a non-negative least squares reconstuction input from measurements. This is to *warm-start* the network as the network need not learn the mapping from measurement domain back to image domain.
- **ProjNets**: These networks are trained to estimate projections on *one* random subspace from given input.
- **SubNet**: This network estimates projections on *mutliple* random subspaces while still maintaining robustness inherent in ProjNets.
- To reconstruct, once we know the estimated projections in these random low-dimensional subspaces we can build a linear system by concatenating the basis vectors of these low-dimensional subpsaces and solving the linear system.

We compare our results with a system that is trained to map the same input to the model directly. We show that our method achieves:
- Robustness to noise and arbitrary deviations from noise model
- Robustness against dataset used for training 

To see all the results, the scheme and the rationale explained in detail, please look at our [paper](https://arxiv.org/pdf/1805.11718.pdf).

## Getting started
1. Prepare training data. 20,000 training images and four geo images are available here: https://uofi.app.box.com/v/deepmesh-data .
2. Use python3 to generate meshes from utils/mesh_code.py . By default this code generates ten 128 x 128 meshes with 50 triangles. 
In our paper we used 130 meshes to train 130 ProjNets and 350 meshes for SubNet.

## Code

### Training multiple ProjNets
To train multiple ProjNets, use ```projnet/train_projnets.py```.
The arguments are as follows:
```console
usage: train_projnets.py [-h] [--imgs IMGS] [--val VAL] [--orig ORIG]
                         [--input INPUT] [--nets NETS] [--path PATH]

optional arguments:
  -h, --help            show this help message and exit
  --imgs IMGS, --images IMGS
                        Number of images to load
  --val VAL, --validation VAL
                        Number of images for validation
  --orig ORIG, --originals ORIG
                        Path to original images .npy
  --input INPUT, --input INPUT
                        Path to ProjNet input images .npy
  --nets NETS, --networks NETS
                        Number of ProjNets to train
  --path PATH, --projnetspath PATH
                        Directory to store ProjNets. Ensure this directory
                        exists.

```

For example, to train 50 ProjNets with 10,000 training images, 100 validation images and save them in ```my_nets``` (ensure this directory exists), we can run the following:
``` console
cd projnet/
python3 train_projnets.py --nets=50 --imgs=10000 --val=10 --path='my_nets' --orig=../originals20k.npy --input=../custom25_infdb.npy
```

## Reconstruct from trained ProjNets
To reconstruct from trained Projnets, use ```projnet/reconstruct_from_projnets.py```. Total variation regularization is used for the reconstruction.
The arguments are as follows:
```console
usage: reconstruct_from_projnets.py [-h] [--orig ORIG] [--input INPUT]
                                    [--nets NETS] [--lam LAM] [--nc]
                                    [--projnets PROJNETS] [--b B] [--c C]
                                    [--path PATH]

optional arguments:
  -h, --help            show this help message and exit
  --orig ORIG, --originals ORIG
                        Path to original images .npy
  --input INPUT, --input INPUT
                        Path to ProjNet input images .npy
  --nets NETS, --networks NETS
                        Number of ProjNets to use
  --lam LAM, --lambda LAM
                        TV regularization parameter
  --nc, --nocoefs       Use if already calculated coefficients
  --projnets PROJNETS, --projnetspath PROJNETS
                        ProjNets directory.
  --b B, --basisstacked B
                        Path to save stacked basis functions .npy
  --c C, --coefsstacked C
                        Path to save stacked coefficients .npy
  --path PATH, --reconpath PATH
                        Reconstruction directory. Ensure this directory
                        exists.
```

For example, to reconstruct from 40 networks in ```my_nets``` with a regularization parameter of 0.003 and store the reconstructions in ```reconstructions```, we can run the following:
```console
cd projnet/
python3 reconstruct_from_projnets.py --nets=40 --projnets=my_nets --lam=0.003 --path=reconstructions --b=basis_40nets.npy --c=coefs_40nets.npy
```

The above command stores the stacked basis functions and stacked coefficients in ```basis.npy``` and ```coefs.npy```.
It is possible that you may wish to try a different regularization parameter for reconstruction. 
As you have saved the stacked basis functions and coefficients, you do not need to calculate these again. You can use the ```--nc``` option:
```console
python3 reconstruct_from_projnets.py --lam=0.002 --path=reconstructions_new_lam --b=basis_40nets.npy --c=coefs_40nets.npy --nc
```
## SubNet, DirectNet

The subspace network (SubNet), takes the basis for the random projection as an input along with the measurements. One can use `subnet/subnet.py` to train the subnet. `subnet.py` allows for resuming training from a particular checkpoint and also, skipping training and moving directly to evaluation on required datasets. We have a common parser for `subnet.py`, `directnet.py` and `reconstruct_from_subnet.py` as they share many same arguments. The parser arguments are as below::

```console
usage: subnet.py [-h] [-ni NITER] [-dnpy DATA_ARRAY] [-mnpy MEASUREMENT_ARRAY]
                 [-ntrain TRAINING_SAMPLES] [-t TRAIN] [-r_iter RESUME_FROM]
                 [-n NAME] [-lr LEARNING_RATE] [-bs BATCH_SIZE]
                 [-i_s IMG_SIZE] [-e EVAL] [-e_orig EVAL_ORIGINALS]
                 [-e_meas EVAL_MEASUREMENTS] [-e_name EVAL_NAME]
                 [-pdir PROJECTORS_DIR] [-nproj NUM_PROJECTORS_TO_USE]
                 [-ntri DIM_RAND_SUBSPACE] [-r_orig RECON_ORIGINALS]
                 [-r_coefs RECON_COEFFICIENTS] [-m MASK]

optional arguments:
  -h, --help            show this help message and exit
  -ni NITER, --niter NITER
                        Number of iterations fot the training to run
  -dnpy DATA_ARRAY, --data_array DATA_ARRAY
                        Numpy array of the data
  -mnpy MEASUREMENT_ARRAY, --measurement_array MEASUREMENT_ARRAY
                        Numpy array of the measurements
  -ntrain TRAINING_SAMPLES, --training_samples TRAINING_SAMPLES
                        Number of training samples
  -t TRAIN, --train TRAIN
                        Training flag
  -r_iter RESUME_FROM, --resume_from RESUME_FROM
                        resume training from r_iter checkpoint, if 0 start
                        training afresh
  -n NAME, --name NAME  Name of directory under results/ where the results of
                        the current run will be stored
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        learning rate to be used for training
  -bs BATCH_SIZE, --batch_size BATCH_SIZE
                        mini-batch size
  -i_s IMG_SIZE, --img_size IMG_SIZE
                        image size
  -e EVAL, --eval EVAL  Evaluation (on other dataset) flag
  -e_orig EVAL_ORIGINALS, --eval_originals EVAL_ORIGINALS
                        list of strings with names of original .npy arrays
  -e_meas EVAL_MEASUREMENTS, --eval_measurements EVAL_MEASUREMENTS
                        list of strings with names of measurement .npy arrays
  -e_name EVAL_NAME, --eval_name EVAL_NAME
                        Name to be given to the evaluation experiments
  -pdir PROJECTORS_DIR, --projectors_dir PROJECTORS_DIR
                        directory where all projector matrices are stored
  -nproj NUM_PROJECTORS_TO_USE, --num_projectors_to_use NUM_PROJECTORS_TO_USE
                        Number of projector matrices to use
  -ntri DIM_RAND_SUBSPACE, --dim_rand_subspace DIM_RAND_SUBSPACE
                        Number of triangles per mesh
  -r_orig RECON_ORIGINALS, --recon_originals RECON_ORIGINALS
                        npy array of originals to compare against
  -r_coefs RECON_COEFFICIENTS, --recon_coefficients RECON_COEFFICIENTS
                        npy array of coefficients to use
  -m MASK, --mask MASK  Mask to apply for convex hull of sensors

```

One must provide the `-dnpy` and `-mnpy` arguments which correspond to the data and the measurements numpy arrays. Along with that, a directory which has all the basis vectors must be provided via `-pdir` argument. Note that running the training does not require you to be in the subnet folder. 

```console
python3 subnet/subnet.py -niter 20000 -dnpy 'originals20k.npy' -mnpy 'custom25_10db.npy' -n test_subnet -e_orig [geo_originals.npy','geo_originals.npy'] -e_meas ['geo_pos_recon_10db.npy','geo_pos_recon_infdb.npy'] -e_name ['geo_tr0_t10','geo_tr0_tinf'] -pdir 'meshes/' -nproj 350 -ntri 50

```

To reconstruct from SubNet, one needs to run `reconstruct_from_subnet.py`. This file takes the coefficients calculated and the projections and runs an iterative projected least squares. Note that since we only train one network for all subspaces we need not use any extra regularization (like TV) to reconstruct the model. An example usage is given below:

```console
python3 subnet/subnet.py -lr 0.0005 -r_orig 'geo_originals.npy' -r_coef 'geo_tr10_tinf' -m 'mask.npy' -nproj 350 -ntri 50
```

## Direct inversion

The direct net uses the same parser as subnet. An example usage is given below for reference:

```console
python3 subnet/directnet.py -niter 20000 -dnpy 'originals20k.npy' -mnpy 'custom25_0db.npy' -n test_subnet -e_orig [geo_originals.npy','geo_originals.npy'] -e_meas ['geo_pos_recon_10db.npy','geo_pos_recon_infdb.npy'] -e_name ['geo_tr0_t10','geo_tr0_tinf']

```

