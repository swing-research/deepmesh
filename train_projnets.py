import keras
import pickle

import projnet
from projnet import make_cnn, compile_fit_cnn, split_npys

def get_P_Pinv(mesh_number):
    path = 'meshes/mesh' + str(mesh_number) + '/'
    print('Getting mesh from: ' + path)
    
    with open(path + 'P.pkl', 'rb') as m:
        P = pickle.load(m)
    with open(path + 'Pinv.pkl', 'rb') as m:
        Pinv = pickle.load(m)
    
    return P, Pinv
    

#########################################       
if __name__ == "__main__":
    # Separate training and validation data
    training_input, test_input, training_truth, test_truth = split_npys(
            10000, original_path='originals20k.npy', 
            input_path='custom25_infdb.npy', val_size=100)
    print('Training input shape: ', training_input.shape)
    print('Test input shape: ', test_input.shape)
    print('Training truth shape: ', training_truth.shape)
    print('Test truth shape: ', test_truth.shape)
    
    number_of_projnets = 20
    root = 'projnets/' # directory in which to store the ProjNets
    
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
        compile_fit_cnn(model, batch_size=50, epochs=3, lr=1e-4, model_name=name, 
                        training_input=training_input, test_input=test_input, 
                        training_truth=training_truth, test_truth=test_truth)
        
        keras.backend.clear_session()
        
        