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
To train multiple ProjNets, use ```projnet/train_projnets.py ```. The arguments are as follows:
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

For example, to train 50 ProjNets with 10,000 training images, 100 validation images and save them in ``` my_nets ```, we can run the following:
``` console
cd projnet/
python train_projnets.py --nets=50 --imgs=10000 --val=10 --path='my_nets' --orig=../originals20k.npy --input=../custom25_infdb.npy
```