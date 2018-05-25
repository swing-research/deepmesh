import keras
import pickle

import sys
sys.path.insert(0, '../utils/')
from train_projnets_parser import train_projnets_parser
import projnet
from projnet import make_cnn, compile_fit_cnn, split_npys

def get_P_Pinv(mesh_number):
    path = '../meshes/mesh' + str(mesh_number) + '/'
    print('Getting mesh from: ' + path)
    
    with open(path + 'P.pkl', 'rb') as m:
        P = pickle.load(m)
    with open(path + 'Pinv.pkl', 'rb') as m:
        Pinv = pickle.load(m)
    
    return P, Pinv

def main():
    args = train_projnets_parser()
    
    # Separate training and validation data
    training_input, test_input, training_truth, test_truth = split_npys(
            args.imgs, original_path=args.orig, 
            input_path=args.input, val_size=args.val)
    print('Training input shape: ' + str(training_input.shape))
    print('Test input shape: ' + str(test_input.shape))
    print('Training truth shape: ' + str(training_truth.shape))
    print('Test truth shape: ' + str(test_truth.shape))
    
    number_of_projnets = args.nets
    root = args.path + '/' # directory in which to store the ProjNets
    print ('Train ' + str(number_of_projnets) + ' ProjNets')
    
    # Train each projnet
    for mesh in range(number_of_projnets):
        print ('Training for mesh: ' + str(mesh))
        
        # Get the basis functions for the mesh and pass them to Projnet
        P, Pinv = get_P_Pinv(mesh)
        projnet.P = P
        projnet.Pinv = Pinv
        
        # Make each ProjNet
        model = make_cnn(channels=32)
        name= root + str(mesh) + '.h5'
        compile_fit_cnn(model, batch_size=50, epochs=25, lr=1e-3, model_name=name, 
                        training_input=training_input, test_input=test_input, 
                        training_truth=training_truth, test_truth=test_truth)
        
        # Manually reinitializing the learning rate helps convergence
        compile_fit_cnn(model, batch_size=50, epochs=3, lr=1e-4, model_name=name, 
                        training_input=training_input, test_input=test_input, 
                        training_truth=training_truth, test_truth=test_truth)
        
        keras.backend.clear_session()

###############################################################################
if __name__ == "__main__":
    main()