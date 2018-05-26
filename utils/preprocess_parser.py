import argparse

def valid_noise(string_value):
    if (string_value=='inf'):
        return string_value
    else:
        return float(string_value)

def preprocess_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--orig', '--originals', help='Path to original images .npy', 
                        type=str, default='../originals20k.npy')
    
    parser.add_argument('--F', '--forwardop', help='Path to forward operator .npy', 
                        type=str, default='../foward_25sensors.npy')

    parser.add_argument('--n', '--numberofimages', help='Number of images to process', 
                        type=int, default=11000)
    
    parser.add_argument('--p', '--numberofprocs', help='Number of processes for multiprocessing', 
                        type=int, default=4)
    
    parser.add_argument('--s', '--SNR', help='Measurement SNR. \'inf\' for no noise. number otherwise',
                        type=valid_noise, default='inf')
    
    parser.add_argument('--d', '--destination', help='Directory to store processed images.'
                        + ' Ensure this directory exists.', 
                        type=str, default='../samples')
    
    parser.add_argument('--r', '--resultnpy', help='Path to final .npy', 
                        type=str, default='../samples.npy')
    
    args = parser.parse_args()
    
    return args
    
###############################################################################
if __name__ == '__main__':
    args = preprocess_parser()
    print(args)
    print (type(args.s))