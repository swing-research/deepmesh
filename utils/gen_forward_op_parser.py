import argparse

def gen_forward_op_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--n', '--nsensors', help='Number of sensors', 
                        type=int, default=25)
    
    parser.add_argument('--g', '--gridsize', help='Grid size', 
                        type=int, default=128)
    
    args = parser.parse_args()
    
    return args
    
###############################################################################
if __name__ == '__main__':
    args = gen_forward_op_parser()
    print(args)