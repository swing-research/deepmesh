import tensorflow as tf
import tensorflow.contrib.layers as tcl


def nn(x, reuse=True, nchstart=32, act_fn=tf.nn.leaky_relu, TRAIN_FLAG=True, REG=False):
    """
    Takes as input the (processed) measurements and 
    estimates a projection of the original image.

    Params
    ------
    x: batch_size, img_size, img_size, nch
    reuse: reuse variables flag
    nchstart: number of output channels in the first convolutional layer
    act_fn: activation function
    REG: Flag to add regularization loss
    TRAIN_FLAG: 'is_training' flag for batch_norm

    Returns
    -------
    out: [batch_size, img_size*img_size] vectors that
    will be struck with the projection for reconstruction

    """
    nchannels = nchstart
    normalizer_params = {'is_training': TRAIN_FLAG}

    reg = tcl.l1_l2_regularizer(scale_l1=1e-4,
                                scale_l2=1e-4)
    reg_loss = 0

    params = {'kernel_size': 3,
              'activation_fn': act_fn,
              'normalizer_fn': tcl.batch_norm,
              'normalizer_params': normalizer_params}

    with tf.variable_scope('projector', reuse=reuse):
        """Downsampling layers"""

        # Block 1
        out1_1 = tcl.conv2d(x, num_outputs=nchannels, **params)
        out1_2 = tcl.conv2d(out1_1, num_outputs=nchannels, **params)
        out_mp1 = tcl.max_pool2d(out1_2, kernel_size=[2, 2], stride=2)

        # Block 2
        out2_1 = tcl.conv2d(out_mp1, num_outputs=2*nchannels, **params)
        out2_2 = tcl.conv2d(out2_1, num_outputs=2*nchannels, **params)
        out_mp2 = tcl.max_pool2d(out2_2, kernel_size=[2, 2], stride=2)

        # Block 3
        out3_1 = tcl.conv2d(out_mp2, num_outputs=4*nchannels, **params)
        out3_2 = tcl.conv2d(out3_1, num_outputs=4*nchannels, **params)
        out_mp3 = tcl.max_pool2d(out3_2, kernel_size=[2, 2], stride=2)

        # Block 4
        out4_1 = tcl.conv2d(out_mp3, num_outputs=8*nchannels, **params)
        out4_2 = tcl.conv2d(out4_1, num_outputs=8*nchannels, **params)
        out_mp4 = tcl.max_pool2d(out4_2, kernel_size=[2, 2], stride=2)

        # Block 5
        out5_1 = tcl.conv2d(out_mp4, num_outputs=16*nchannels, **params)
        out5_2 = tcl.conv2d(out5_1, num_outputs=16*nchannels, **params)

        # regularization
        if REG:
            reg_loss = reg(tcl.flatten(out5_2))

        """Upsampling layers"""

        # Block 1
        up_out1_1 = tf.keras.layers.UpSampling2D((2, 2))(out5_2)
        up_out1_1 = tf.concat([out4_2, up_out1_1], axis=3, name='skip_1')
        up_out1_1 = tcl.conv2d(up_out1_1, num_outputs=8*nchannels, **params)
        up_out1_2 = tcl.conv2d(up_out1_1, num_outputs=8*nchannels, **params)

        # Block 2
        up_out2_1 = tf.keras.layers.UpSampling2D((2, 2))(up_out1_2)
        up_out2_1 = tf.concat([out3_2, up_out2_1], axis=3, name='skip_2')
        up_out2_1 = tcl.conv2d(up_out2_1, num_outputs=4*nchannels, **params)
        up_out2_2 = tcl.conv2d(up_out2_1, num_outputs=4*nchannels, **params)

        # Block 3
        up_out3_1 = tf.keras.layers.UpSampling2D((2, 2))(up_out2_2)
        up_out3_1 = tf.concat([out2_2, up_out3_1], axis=3, name='skip_3')
        up_out3_1 = tcl.conv2d(up_out3_1, num_outputs=2*nchannels, **params)
        up_out3_2 = tcl.conv2d(up_out3_1, num_outputs=2*nchannels, **params)

        # Block 4
        up_out4_1 = tf.keras.layers.UpSampling2D((2, 2))(up_out3_2)
        up_out4_1 = tf.concat([out1_2, up_out4_1], axis=3, name='skip_4')
        up_out4_1 = tcl.conv2d(up_out4_1, num_outputs=nchannels, **params)
        up_out4_2 = tcl.conv2d(up_out4_1, num_outputs=nchannels, **params)

        # Block 5
        up_out5_1 = tcl.conv2d(up_out4_2, num_outputs=1, **params)

        out = tf.contrib.layers.flatten(up_out5_1)

    return out, reg_loss
