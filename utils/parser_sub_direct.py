import argparse


def parserMethod():
    parser = argparse.ArgumentParser()

    parser.add_argument("-ni", "--niter", type=int, default=20000,
                        help="Number of iterations fot the training to run")
    parser.add_argument("-dnpy", "--data_array", type=str,
                        default='./../originals20k.npy', help='Numpy array of the data')
    parser.add_argument("-mnpy", "--measurement_array", type=str,
                        default='./../custom25_infdb.npy', help='Numpy array of the measurements')
    parser.add_argument("-ntrain", "--training_samples", type=int,
                        default=19500, help='Number of training samples')
    parser.add_argument("-t", "--train", type=bool,
                        default=True, help="Training flag")
    parser.add_argument('-r_iter', '--resume_from', type=int,
                        default=0, help='resume training from r_iter checkpoint,' +
                        ' if 0 start training afresh')
    parser.add_argument("-n", "--name", type=str, default='trial',
                        help='Name of directory under results/ where ' +
                        'the results of the current run will be stored')
    parser.add_argument("-lr", "--learning_rate", type=float, default=5e-3,
                        help='learning rate to be used for training')
    parser.add_argument("-bs", "--batch_size", type=int, default=50,
                        help='mini-batch size')

    # eval specific arguments
    parser.add_argument("-e", "--eval", type=bool,
                        default=True, help="Evaluation (on other dataset) flag")
    parser.add_argument("-e_orig", "--eval_originals", type=list,
                        default=[], help="list of strings with names of original .npy arrays")
    parser.add_argument("-e_meas", "--eval_measurements", type=list,
                        default=[], help="list of strings with names of measurement .npy arrays")
    parser.add_argument("-e_name", "--eval_name", type=list,
                        default=[], help="Name to be given to the evaluation experiments")


    # subnet specific arguments
    parser.add_argument('-pdir', '--projectors_dir',
                        default='./../meshes/',
                        type=str, help='directory where all projector matrices are stored')
    parser.add_argument('-nproj', '--num_projectors_to_use',
                        default=350, type=int, help='Number of projector matrices to use')
    parser.add_argument('-ntri', '--dim_rand_subspace',
                        default=50, type=int, help='Number of triangles per mesh')

    args = parser.parse_args()

    return args


def test(args):
    for k, v in args.__dict__.items():
        print('%s: %s' % (k, str(v)))

    return


if __name__ == '__main__':
    parserMethod()
