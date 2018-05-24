import numpy as np
from scipy import misc
import multiprocessing as mp

IMAGE_DIM = 128
 
def multiproc_custom_transform():
    F = np.load('forward_25circled.npy')
    
    number_of_imgs = 20000
    images = np.load('originals20k.npy')    
    print ('loaded images')
    
    nprocs = min(2, number_of_imgs)
    pool = mp.Pool(processes=nprocs)
    
    results = [pool.apply_async(positive_custom_transform_worker,
                                args=(images[i,:,:,0], F, i)) for i in range(0, 12)]

    for p in results:
        p.get()

    return None
        
def positive_custom_transform_worker(img, F, im_num):
    np.random.seed()
    lr = 1e-3
    n_iter = 300
    img = img.reshape((-1,1))
    y = F.dot(img)
    noise_std = calc_noise_std(y, 10)
    y += np.random.normal(scale=noise_std, size=y.shape)
    x = np.zeros(img.shape)  

    for i in range(n_iter):
        x += lr* ((F.T).dot(y - F.dot(x)))
        x[x<0]=0
        x[x>1]=1
    
    filename = 'custom_transform_25_circ_positivity_infdB/sample_' + str(im_num) + '.png'
    misc.imsave(filename, x.reshape((128,128)))
    if (im_num%10==0):
        print (im_num)
    
    return None    

def save_npys(number_of_images):
#    originals = np.zeros([number_of_images, IMAGE_DIM, IMAGE_DIM, 1])
    cnn_input = np.zeros([number_of_images, IMAGE_DIM, IMAGE_DIM, 1])
    
    for i in range(number_of_images):
#        original_filename = 'originals/image_' + str(i) + '.png'
#        originals[i,:,:,0] = misc.imread(original_filename) / 255.0
        
#        cnn_input_filename = 'custom_transform_25_circ_positivity_infdB/sample_' + str(i) + '.png'
        cnn_input_filename = 'random_samples/128/sample_' + str(i) + '.png'
        cnn_input[i,:,:,0] = misc.imread(cnn_input_filename) / 255.0
        
        if (i%100==0):
            print (i)
        
#    np.save('originals20k.npy', originals)
#    np.save('custom25_infdb.npy', cnn_input)
    np.save('random_samples/random_samples_infdb.npy', cnn_input)

def calc_noise_std(obs, desiredDBsnr):
    obs_var = (np.std(obs))**2
    noise_var = obs_var / (10**(desiredDBsnr/10))
    return noise_var**0.5

if __name__ == "__main__":
#    number_of_images = 4
#    originals = import_data(number_of_images)
#    np.save('originals_noscale.npy', originals)
#    fbp_images = do_radon(originals, 5)
#    np.save('originals_fbp_5_noscale.npy', fbp_images)
#    
#    
#    originals = np.load('originals_noscale.npy')
#    fbps = np.load('originals_fbp_5_noscale.npy')
#    projected1 = proj_using_tf(fbps)
#    np.save('projected_originals_noscale.npy', projected)
    
#    originals_train, _, _, fbps_train, _, _ = get_all_data(4, 0)
    
#    proj_pngs_using_tf(range(7080, 30000))
    
#    do_radon_ellipse()
    
#    custom_forward_operator()

#    positivity_custom_transform_on_one()
    
#    positivity_custom_transform()
    
#    multiproc_custom_transform()
    
#    save_npys(12)
    
#    get_face()
#    positivity_custom_transform_on_one()
    
#    do_FBP_on_shapes()
    
#    generate_boards()
    
#    blocks = get_blocks()
    
#    random_samples()
    
    pass