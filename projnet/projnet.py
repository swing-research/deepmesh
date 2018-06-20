from __future__ import print_function
import keras
import keras.backend as K
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, UpSampling2D, Conv2DTranspose, BatchNormalization, Lambda, Input
from keras.models import Model

import numpy as np

import sys
sys.path.insert(0, '../utils/')
from preprocess import IMAGE_DIM

# These are overwritten by matrices when the Projnet is called.
P = 0
Pinv = 0

def separate_training_from_val(data, number_of_images, test_size=100):
    training = data[:number_of_images-test_size, :,:,:]
    truth = data[number_of_images-test_size:, :,:,:]
    return training, truth

def split_npys(number_of_images, original_path, input_path, val_size=100):
    originals = np.load(original_path)
    originals = originals[:number_of_images]
    print ('loaded originals: ', original_path)
    training_truth, test_truth = separate_training_from_val(originals, number_of_images, test_size=val_size)
    del originals    
    
    cnn_input = np.load(input_path)
    cnn_input = cnn_input[:number_of_images]
    print ('loaded cnn_input: ', input_path)
    training_input, test_input = separate_training_from_val(cnn_input, number_of_images, test_size=val_size)
    del cnn_input
    
    return training_input.astype('float32'), test_input.astype('float32'), training_truth.astype('float32'), test_truth.astype('float32')    

def projection_loss(yTrue, yPred):
    # This method calculates the loss between the projection of an image and
    # the ProjNet output.
    # The inputs are the original image and the CNN output.
    
    # First the basis functions are applied to the original images to get 
    # coefficients.
    yTrue = keras.backend.batch_flatten(yTrue)
    Pt = P.T
    A = K.variable(Pt)
    c = K.dot(yTrue,A)    
    
    # Secondly, the pseudoinverse of the basis function is applied to the
    # coefficients to obtain projections.
    Pinvt = Pinv.T
    A = K.variable(Pinvt)
    out = K.dot(c,A)
    out = K.reshape(out, (-1,IMAGE_DIM,IMAGE_DIM,1))
    
    return keras.losses.mean_squared_error(out, yPred)

def apply_P(x):
    # This method applied the basis function to the output of the previous 
    # CNN layer to get coefficients.
    # The input is the output of the CNN. The output of this method is passed 
    # to apply_Pinv().
    
    import keras
    
    x = keras.backend.batch_flatten(x)
    Pt = P.T
    A = K.variable(Pt)
    out = K.dot(x,A)
    return out
        
def apply_Pinv(x):
    # This method applies the pseudoinverse of the basis functions to get an
    # estimate of the projection of the original image.
    
    import keras
     
    Pinvt = Pinv.T
    A = K.variable(Pinvt)
    out = K.dot(x,A)
    out = K.reshape(out, (-1,IMAGE_DIM,IMAGE_DIM,1))
    return out

def make_cnn(channels):
    # This method creates a ProjNet model.
    
    filter_size = 3
    filter_padding = int((filter_size -1)/2)
    input_shape = (IMAGE_DIM, IMAGE_DIM, 1)
    
    nn_input = Input(shape=input_shape)
    
    zero_pad1_1 = ZeroPadding2D(padding=filter_padding, data_format=None)(nn_input)
    conv1_1 = Conv2D(channels, kernel_size=filter_size, strides=(1, 1), 
                     activation='relu')(zero_pad1_1)
    batch1_1 = BatchNormalization()(conv1_1)
    
    zero_pad1_3 = ZeroPadding2D(padding=filter_padding, data_format=None)(batch1_1)
    conv1_3 = Conv2D(channels, kernel_size=filter_size, activation='relu')(zero_pad1_3)
    batch1_3 = BatchNormalization()(conv1_3)
    pool1_3 = MaxPooling2D(pool_size=(2, 2))(batch1_3)
    
    zero_pad2_1 = ZeroPadding2D(padding=filter_padding, data_format=None)(pool1_3)
    conv2_1 = Conv2D(channels*2, kernel_size=filter_size, activation='relu')(zero_pad2_1)
    batch2_1 = BatchNormalization()(conv2_1)
    
    zero_pad2_3 = ZeroPadding2D(padding=filter_padding, data_format=None)(batch2_1)
    conv2_3 = Conv2D(channels*2, kernel_size=filter_size, activation='relu')(zero_pad2_3)
    batch2_3 = BatchNormalization()(conv2_3)
    pool2_3 = MaxPooling2D(pool_size=(2, 2))(batch2_3)
    
    zero_pad3_1 = ZeroPadding2D(padding=filter_padding, data_format=None)(pool2_3)
    conv3_1 = Conv2D(channels*4, kernel_size=filter_size, activation='relu')(zero_pad3_1)
    batch3_1 = BatchNormalization()(conv3_1)
    
    zero_pad3_3 = ZeroPadding2D(padding=filter_padding, data_format=None)(batch3_1)
    conv3_3 = Conv2D(channels*4, kernel_size=filter_size, activation='relu')(zero_pad3_3)
    batch3_3 = BatchNormalization()(conv3_3)
    pool3_3 = MaxPooling2D(pool_size=(2, 2))(batch3_3)
    
    zero_pad4_1 = ZeroPadding2D(padding=filter_padding, data_format=None)(pool3_3)
    conv4_1 = Conv2D(channels*8, kernel_size=filter_size, activation='relu')(zero_pad4_1)
    batch4_1 = BatchNormalization()(conv4_1)
    
    zero_pad4_3 = ZeroPadding2D(padding=filter_padding, data_format=None)(batch4_1)
    conv4_3 = Conv2D(channels*8, kernel_size=filter_size, activation='relu')(zero_pad4_3)
    batch4_3 = BatchNormalization()(conv4_3)
    pool4_3 = MaxPooling2D(pool_size=(2, 2))(batch4_3)
    
    zero_pad5_1 = ZeroPadding2D(padding=filter_padding, data_format=None)(pool4_3)
    conv5_1 = Conv2D(channels*16, kernel_size=filter_size, activation='relu')(zero_pad5_1)
    batch5_1 = BatchNormalization()(conv5_1)
    
    zero_pad5_3 = ZeroPadding2D(padding=filter_padding, data_format=None)(batch5_1)
    conv5_3 = Conv2D(channels*16, kernel_size=filter_size, activation='relu')(zero_pad5_3)
    batch5_3 = BatchNormalization()(conv5_3)
   
    upsample7_3 = UpSampling2D(size=(2, 2), data_format=None)(batch5_3)
    merged_4 = keras.layers.concatenate([batch4_3, upsample7_3])
    zero_pad8_1 = ZeroPadding2D(padding=filter_padding, data_format=None)(merged_4)
    conv8_1 = Conv2D(channels*8, kernel_size=filter_size, activation='relu')(zero_pad8_1)
    batch8_1 = BatchNormalization()(conv8_1)
    
    zero_pad8_3 = ZeroPadding2D(padding=filter_padding, data_format=None)(batch8_1)
    conv8_3 = Conv2D(channels*8, kernel_size=filter_size, activation='relu')(zero_pad8_3)
    batch8_3 = BatchNormalization()(conv8_3)
   
    upsample8_3 = UpSampling2D(size=(2, 2), data_format=None)(batch8_3)   
    merged_3 = keras.layers.concatenate([batch3_3, upsample8_3])
    zero_pad9_1 = ZeroPadding2D(padding=filter_padding, data_format=None)(merged_3)
    conv9_1 = Conv2D(channels*4, kernel_size=filter_size, activation='relu')(zero_pad9_1)
    batch9_1 = BatchNormalization()(conv9_1)
    
    zero_pad9_3 = ZeroPadding2D(padding=filter_padding, data_format=None)(batch9_1)
    conv9_3 = Conv2D(channels*4, kernel_size=filter_size, activation='relu')(zero_pad9_3)
    batch9_3 = BatchNormalization()(conv9_3)
    
    upsample9_3 = UpSampling2D(size=(2, 2), data_format=None)(batch9_3)
    merged_2 = keras.layers.concatenate([batch2_3, upsample9_3])
    zero_pad10_1 = ZeroPadding2D(padding=filter_padding, data_format=None)(merged_2)
    conv10_1 = Conv2D(channels*2, kernel_size=filter_size, activation='relu')(zero_pad10_1)
    batch10_1 = BatchNormalization()(conv10_1)
    
    
    zero_pad10_2 = ZeroPadding2D(padding=filter_padding, data_format=None)(batch10_1)
    conv10_2 = Conv2D(channels*2, kernel_size=filter_size, activation='relu')(zero_pad10_2)
    batch10_2 = BatchNormalization()(conv10_2)
    
    upsample10_3 = UpSampling2D(size=(2, 2), data_format=None)(batch10_2)
    merged_1 = keras.layers.concatenate([batch1_3, upsample10_3])
    zero_pad11_1 = ZeroPadding2D(padding=filter_padding, data_format=None)(merged_1)
    conv11_1 = Conv2D(channels, kernel_size=filter_size, activation='relu')(zero_pad11_1)
    batch11_1 = BatchNormalization()(conv11_1)
    
    conv11_3 = Conv2D(channels, kernel_size=filter_size, activation='relu')(batch11_1)
    batch11_3 = BatchNormalization()(conv11_3)
    
    conv_2D_transpose = Conv2DTranspose(1, filter_size, activation='relu')(batch11_3)
    
    # Lambda layers perform matrix multiplication to project layer input onto mesh
    lambda1 = Lambda(apply_P)(conv_2D_transpose)
    output = Lambda(apply_Pinv)(lambda1)
       
    model = Model(inputs=nn_input, outputs=output)
    
    return model
    
def compile_fit_cnn(model, batch_size, epochs, lr, model_name,
                    training_input, test_input, training_truth, test_truth):
    # This method compiles the CNN and trains it.
    
    model.compile(loss=projection_loss,
                  optimizer=keras.optimizers.Adam(lr=lr),
                  metrics=['mse'])
    
    class AccuracyHistory(keras.callbacks.Callback):
        def on_train_begin(self, logs={}):
            self.mse = []
    
        def on_epoch_end(self, batch, logs={}):
            self.model.save(model_name)
    
    history = AccuracyHistory()
    
    model.fit(training_input, training_truth,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(test_input, test_truth),
              callbacks=[history], shuffle=True)
    score = model.evaluate(test_input, test_truth, verbose=0)
    
    return model, score, history
    

###############################################################################     
if __name__ == "__main__":
    pass
 