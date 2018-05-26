import numpy as np
from scipy import misc
import multiprocessing as mp

from preprocess_parser import preprocess_parser

IMAGE_DIM = 128
 
def multiproc_custom_transform():
    args = preprocess_parser()
    
    F = np.load(args.F)
    
    number_of_imgs = args.n
    images = np.load(args.orig)    
    print ('loaded originals')
    
    nprocs = min(args.p, number_of_imgs)
    pool = mp.Pool(processes=nprocs)
    
    results = [pool.apply_async(positive_custom_transform_worker,
                                args=(images[i,:,:,0], F, i)) for i in range(0, number_of_imgs)]

    for p in results:
        p.get()

    return None
        
def positive_custom_transform_worker(img, F, im_num):
    args = preprocess_parser()
    
    np.random.seed()
    
    noise = args.s
    lr = 1e-3
    n_iter = 300
    if (type(noise)==float and noise<=10):
        lr = 1e-4
    
    img = img.reshape((-1,1))
    y = F.dot(img)
    if (noise != 'inf'):
        noise_std = calc_noise_std(y, noise)
        y += np.random.normal(scale=noise_std, size=y.shape)
    x = np.zeros(img.shape)  

    for i in range(n_iter):
        x += lr* ((F.T).dot(y - F.dot(x)))
        x[x<0]=0
        x[x>1]=1
    
    filename = args.d + '/sample_' + str(im_num) + '.png'
    misc.imsave(filename, x.reshape((128,128)))
    if (im_num%100==0):
        print ('Number of images complete: ' + str(im_num))
    
    return None

def calc_noise_std(obs, desiredDBsnr):
    obs_var = (np.std(obs))**2
    noise_var = obs_var / (10**(desiredDBsnr/10))
    return noise_var**0.5

def save_npys():
    args = preprocess_parser()
    
    number_of_images = args.n
    cnn_input = np.zeros([number_of_images, IMAGE_DIM, IMAGE_DIM, 1])
    
    for i in range(number_of_images):
        cnn_input_filename = args.d + '/sample_' + str(i) + '.png'
        cnn_input[i,:,:,0] = misc.imread(cnn_input_filename) / 255.0
        
        if (i%100==0):
            print ('Number of images saved to .npy: ' + str(i))

    np.save(args.r, cnn_input)
    return None

def main():
    multiproc_custom_transform()
    save_npys()
    return None

if __name__ == "__main__":
    main()