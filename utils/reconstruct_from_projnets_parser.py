import argparse

def reconstruct_from_projnets_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--orig', '--originals', help='Path to original images .npy', 
                        type=str, default='../geo_originals.npy')
    
    parser.add_argument('--input', '--input', help='Path to ProjNet input images .npy', 
                        type=str, default='../geo_pos_recon_infdb.npy')
    
    parser.add_argument('--nets', '--networks', help='Number of ProjNets to use', 
                        type=int, default=20)
    
    parser.add_argument('--lam', '--lambda', help='TV regularization parameter', 
                        type=int, default=4e-3)
    
    parser.add_argument('--nc', '--nocoefs', help='Use if already calculated coefficients',
                        action='store_false')

    parser.add_argument('--projnets', '--projnetspath', help='ProjNets directory.', 
                        type=str, default='projnets')
    
    parser.add_argument('--b', '--basisstacked', help='Path to save stacked basis functions .npy', 
                        type=str, default='B_stacked.npy')
    
    parser.add_argument('--c', '--coefsstacked', help='Path to save stacked coefficients .npy', 
                        type=str, default='coefs_stacked.npy')
    
    parser.add_argument('--path', '--reconpath', help='Reconstruction directory.'
                        + ' Ensure this directory exists.', 
                        type=str, default='reconstructions')
    
    args = parser.parse_args()
    
    return args
    
###############################################################################
if __name__ == '__main__':
    args = reconstruct_from_projnets_parser()
    print(args)