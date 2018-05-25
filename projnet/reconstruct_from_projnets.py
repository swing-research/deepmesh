import keras
from keras.models import Model
import numpy as np
from scipy import misc
import matplotlib as mpl
mpl.use('Agg')

import sys
sys.path.insert(0, '../utils/')
from preprocess import IMAGE_DIM
from SNRab import SNRab
from train_projnets import get_P_Pinv
import projnet
from projnet import projection_loss

NUMBER_OF_COEFS = 50 # Number of basis functions

def get_coefs(main_dir, net, test_input, P, Pinv):
    model_name = main_dir + str(net)
    print ('getting: ', model_name + '.h5')
    
    # Load each ProjNet
    model = keras.models.load_model(model_name + '.h5', custom_objects=
                                    {'projection_loss': projection_loss,
                                     'P': P, 'Pinv': Pinv, 
                                     'IMAGE_DIM': IMAGE_DIM})
    print ('got: ' + model_name + '.h5')
    
    # Taking the output from the second last layer is taking the output from 
    # the first lambda layer in the ProjNet model. This will be the
    # coefficients. Therefore we build a model which is ProjNet without the 
    # final layer (which is also the second lambda layer)
    coef_model = Model(inputs=model.input, outputs=model.layers[-2].output)
    coefs = coef_model.predict(test_input)
    
    del model
    del coef_model
    
    return coefs
    
def get_stacked_P_and_coefs(main_dir, number_of_networks, test_input):
    # Get the stacked basis function matrices and coefficients
    
    P_stacked = np.zeros([NUMBER_OF_COEFS*number_of_networks, IMAGE_DIM**2])
    coefs = np.zeros([test_input.shape[0], NUMBER_OF_COEFS*number_of_networks])
    
    # Run the test images through each ProjNet and get the coeffecients
    for i in range(number_of_networks):
        P, Pinv = get_P_Pinv(i)
        projnet.P = P
        projnet.Pinv = Pinv
        c = get_coefs(main_dir, i, test_input, P, Pinv)
        keras.backend.clear_session()
        
        # Store coefficients and basis functions
        P_stacked[i*NUMBER_OF_COEFS : (i+1)*NUMBER_OF_COEFS] = P
        coefs[:, i*NUMBER_OF_COEFS : (i+1)*NUMBER_OF_COEFS] = c
        
    return P_stacked, coefs.T

def reconstruct_images_TV(P_stacked, coefs, lam):
    # Reconstruct using TV regularization
    # We use the proxTV toolbox for this: https://github.com/albarji/proxTV
    
    import prox_tv as ptv
    lr = 0.5
    n_iter = 400
    
    x = np.zeros([IMAGE_DIM**2, coefs.shape[1]])
    for i in range(n_iter):
        delta = coefs - P_stacked.dot(x)
        x += lr*(P_stacked.T).dot(delta)
        for img in range(coefs.shape[1]):
            # Do each image separately
            x[:,img] = (ptv.tv1_2d(x[:,img].reshape((IMAGE_DIM, IMAGE_DIM)), 
             lam*lr)).flatten()
        
        # Each image should be between 0 and 1 so we enforce this constraint
        x[x<0] =0
        x[x>1] =1
        
        if (i%100==0):
            print ('Number of iterations: ' + str(i))
    
    # Some code to manipulate and organize the reconstructions
    recons = np.zeros([coefs.shape[1], IMAGE_DIM, IMAGE_DIM])
    for i in range(coefs.shape[1]):
        vec_img = x[:,i]
        recon = vec_img.reshape((IMAGE_DIM, IMAGE_DIM))
        recons[i,:,:] = recon
    
    return recons

#def SNRab(x, x_hat, db=True):
#    # better safe than sorry (matrix vs array)
#    xx = x.flatten()
#    yy = x_hat.flatten()
#    
#    u = xx.sum()
#    v = (xx*yy).sum()
#    w = (yy**2).sum()
#    p = yy.sum()
#    q = len(xx)**2
#    
#    a = (v*q - u*p)/(w*q - p*p)
#    b = (w*u - v*p)/(w*q - p*p)
#    
#    SNRopt = np.sqrt((xx**2).sum() / ((xx - (a*yy + b))**2).sum())
#    SNRraw = np.sqrt((xx**2).sum() / ((xx - yy)**2).sum())
#    
#    if db:
#        SNRopt = 20*np.log10(SNRopt)
#        SNRraw = 20*np.log10(SNRraw)
#
#    return SNRopt, SNRraw, a, b

def save_results(originals, test_input, recons, lam, file_info):
    # Saves the reconstructions as .png and .npy files.
    # Three images are tiled together. From left to right, they are the 
    # original, the ProjNet input and the reconstruction.
    
    # Our forward operator consists of sensors placed unifromly in an inscribed
    # circle, so we are interested in the reconstruction in that region. We 
    # have a mask to zero the regions outside the sensor coverage.
    mask = (np.load('../mask.npy')).reshape((IMAGE_DIM, IMAGE_DIM))
    
    snrs = np.zeros(originals.shape[0])
    
    for i in range(originals.shape[0]):
        truth = originals[i,:,:,0]*mask
        result = recons[i,:,:]*mask
        
        # Tile the three images
        results = np.zeros([IMAGE_DIM, IMAGE_DIM*3])
        results[:,:IMAGE_DIM] = truth
        results[:,IMAGE_DIM:2*IMAGE_DIM] = test_input[i,:,:,0]
        results[:,2*IMAGE_DIM:] = result
                
        # Calcualte some useful SNR information
        snr, _, _, _ = SNRab(truth, result)
        input_snr, _, _, _ = SNRab(truth, test_input[i,:,:,0])
        print ('lam: ' + str(lam) +  ' - im' + str(i+1) + ' - Recon SNR: ' + 
               str(snr) + ' - Input SNR: ' + str(input_snr))
        snrs[i] = snr
        
        # Save as .png and .npy array
        misc.imsave('reconstructions/''im' + str(i+1) + '_' + file_info + 
                    '.png', results)
        np.save('reconstructions/''im' + str(i+1) + '_' + file_info + '.npy',
                result)
    
    print ('AVERAGE SNR FOR lam: ' + str(lam) + ' is ' + str(np.mean(snrs)))

#########################################
if __name__ == "__main__":    
    # load the test images and their constrained least squares reconstruction.
    originals = np.load('../geo_originals.npy')
    test_input = np.load('../geo_pos_recon_infdb.npy')
    
    main_dir = 'projnets/' # directory where ProjNets are stored
    networks = 20 # number of ProjNets to reconstruct from
    
    # Do we need to get stacked basis functions and coefficients?
    coefs_required = True
    if (coefs_required == True):
        # Get the stacked basis function matrices and coefficients
        P_stacked, coefs = get_stacked_P_and_coefs(main_dir, networks, 
                                                   test_input)
    
        # Save stacked basis function matrices and coefficients for later use
        np.save('P_stacked.npy', P_stacked)
        np.save('coefs.npy', coefs)
    
    # Load stacked basis function matrices and coefficients
    P_stacked = np.load('P_stacked.npy')
    P_stacked = P_stacked[:networks*NUMBER_OF_COEFS]
    coefs = np.load('coefs.npy')
    coefs = coefs[:networks*NUMBER_OF_COEFS]
    
    # We have to find x that satisfies the equation: P_stacked x = coefs
    # We can use total variation regularization to do this
    lam = 4e-3 # TV regularization parameter. Found from some holdout set.
    reconstructions = reconstruct_images_TV(P_stacked, coefs, lam)
    save_results(originals, test_input, reconstructions, lam, 'noise10dB')