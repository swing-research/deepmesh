import argparse

def train_projnets_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--imgs', '--images', help='Number of images to load', 
                        type=int, default=19600)
    
    parser.add_argument('--val', '--validation', help='Number of images for validation', 
                        type=int, default=100)
    
    parser.add_argument('--orig', '--originals', help='Path to original images .npy', 
                        type=str, default='../originals20k.npy')
    
    parser.add_argument('--input', '--input', help='Path to ProjNet input images .npy', 
                        type=str, default='../custom25_infdb.npy')
    
    parser.add_argument('--nets', '--networks', help='Number of ProjNets to train', 
                        type=int, default=20)
    
    parser.add_argument('--path', '--projnetspath', help='DIrectory to store ProjNets.'
                        + ' Ensure this directory exists.', 
                        type=str, default='projnets')
    
    args = parser.parse_args()
    
    return args
    
def print_args(args):
    print(args)
    
###############################################################################
if __name__ == '__main__':
    args = train_projnets_parser()
    print_args(args)