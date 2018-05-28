# Deepmesh
Repository for 'Deep Mesh Projectors for Inverse Problems'

We intend to make it as simple as possible to reproduce our results. If something is missing or you would like some guidance, please reach out to us.

## Skeleton of README (tentative)
- Idea, equation (9) from paper
- Explain ProjNet (how to run included here)
- Motivate SubNet (how to run included here)
- Explain reconstruction scheme (TV no TV, PLS)
- Results

## Training multiple ProjNets
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
python train_projnets.py --nets=50 --imgs=10000 --val=10 --path='my_nets' --orig=../originals20k.npy --input=../custom25_infdb.npy
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
python reconstruct_from_projnets.py --nets=40 --projnets=my_nets --lam=0.003 --path=reconstructions --b=basis_40nets.npy --c=coefs_40nets.npy
```

The above command stores the stacked basis functions and stacked coefficients in ```basis.npy``` and ```coefs.npy```.
It is possible that you may wish to try a different regularization parameter for reconstruction. 
As you have saved the stacked basis functions and coefficients, you do not need to calculate these again. You can use the ```--nc``` option:
```console
python reconstruct_from_projnets.py --lam=0.002 --path=reconstructions_new_lam --b=basis_40nets.npy --c=coefs_40nets.npy --nc
```