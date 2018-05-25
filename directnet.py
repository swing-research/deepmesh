import os
import time
import numpy as np
from model import net
from pickle import load
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.misc import imsave, imread
import tensorflow.contrib.layers as tcl


def make_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def summary(name, tensor):
    tf.summary.scalar(name, tensor)
    return tensor


class DNN():
    """
    Implements the CNN for tomographic reconstruction
    on a coarse mesh. Different meshes would be different
    instantiations of the same class.

    """
    def __init__(self, **kwargs):
        """Initializes subnet and loads in all required parameters

        -- creates the required directories for storing results
        -- creates appropriate placeholders for the tf.Graph()
        -- builds the graph

        """
        self.lr = kwargs['lr']
        self.niters = kwargs['niters']
        self.img_ch = kwargs['img_ch']
        self.nbatch = kwargs['nbatch']
        self.ntrain = kwargs['ntrain']
        self.LOG_DIR = kwargs['LOG_DIR']
        self.img_size = kwargs['img_size']
        self.data_npy = kwargs['data_npy']
        self.meas_npy = kwargs['meas_npy']

        make_if_not_exist(self.LOG_DIR)
        make_if_not_exist(self.LOG_DIR+'/train_images')
        make_if_not_exist(self.LOG_DIR+'/test_images')

        # reset default graph
        tf.reset_default_graph()

        # placeholders:
        self.true_img = tf.placeholder(tf.float32,
                                       [None, self.img_size,
                                        self.img_size, self.img_ch])

        self.measurements = tf.placeholder(tf.float32,
                                           [None, self.img_size,
                                            self.img_size, self.img_ch])

        self.TRAIN_FLAG = tf.placeholder(tf.bool)

        # read data
        self.read_data()

        # build the net
        self.build_model()

        return

    def read_data(self):
        """Reads in the data"""
        data = np.load(self.data_npy).reshape(
            -1, self.img_size, self.img_size, 1).astype('float32')
        self.data_train = data[:self.ntrain]
        self.data_test = data[self.ntrain:]

        measurements = np.load(self.meas_npy).reshape(
            -1, self.img_size, self.img_size, 1).astype('float32')

        self.measurements_train = measurements[:self.ntrain]
        self.measurements_test = measurements[self.ntrain:]

        self.nsamples_train = len(self.data_train)
        self.nsamples_test = len(self.data_test)

        print('Data read', flush=True)
        print('Training samples: %d' % (self.nsamples_train), flush=True)
        print('Testing samples: %d' % (self.nsamples_test), flush=True)

        return

    def model(self):
        """
        It assumes that self.x is already in the required shape 
        (-1, img_size, img_size, img_ch)
        """

        # the forward model
        out, reg_loss = nn(self.measurements,
                           reuse=False,
                           TRAIN_FLAG=self.TRAIN_FLAG,
                           nchstart=64,
                           act_fn=tf.nn.leaky_relu)

        return out, reg_loss

    def build_model(self):
        self.out, self.reg_loss = self.model()

        self.out = tf.clip_by_value(self.out, 0.0, 1.0, name='final_clip')

        # get projection of actual image
        flat_out = tcl.flatten(self.true_img)

        """Loss function"""
        self.loss = tf.reduce_mean(
            tf.square(self.out - flat_out)) + self.reg_loss

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.NNtrain = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        summary("nn_loss", self.loss)

        self.summary_op = tf.summary.merge_all()

        return None

    def train(self, rerun=False, ckpt_offset=0):
        t = time.time()

        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.LOG_DIR, sess.graph)

            if rerun:
                saver.restore(sess, tf.train.latest_checkpoint(self.LOG_DIR))

            print('Beginning training now.')

            for i in range(self.niters):

                if (i+1) % 2000 == 0:
                    """Validation loss loop + CKPT loop"""
                    saver.save(sess, self.LOG_DIR+"/model.ckpt", i+ckpt_offset)

                    idx = np.random.choice(np.arange(self.nsamples_test),
                                           self.nbatch,
                                           replace=False)
                    batch = self.data_test[idx]
                    measurements = self.measurements_test[idx]

                    feed_dict_in = {self.true_img: batch,
                                    self.TRAIN_FLAG: False,
                                    self.measurements: measurements}

                    nnloss = sess.run(self.loss, feed_dict=feed_dict_in)

                    print('Validation loss at %d iterations = %f' %
                          (i, nnloss), flush=True)

                    summary = sess.run(self.summary_op, feed_dict=feed_dict_in)
                    summary_writer.add_summary(summary, i+ckpt_offset)

                """Training loss loop"""

                idx = np.random.choice(np.arange(self.nsamples_train),
                                       self.nbatch,
                                       replace=False)
                batch = self.data_train[idx]
                measurements = self.measurements_train[idx]

                feed_dict_in = {self.true_img: batch,
                                self.TRAIN_FLAG: True,
                                self.measurements: measurements}

                _ = sess.run(self.NNtrain, feed_dict=feed_dict_in)

                if (i+1) % 1000 == 0:
                    loss, y, out = sess.run(
                        [self.loss, self.measurements, self.out],
                        feed_dict=feed_dict_in)

                    print("Training loss at epoch %d = %f" %
                          (i+ckpt_offset, loss), flush=True)

                    imsave(self.LOG_DIR+'/train_images/y_%d.png' %
                           (i+ckpt_offset), y[:25].reshape(
                               5, 5, 128, 128).swapaxes(
                               1, 2).reshape(5*128, -1))
                    imsave(self.LOG_DIR+'/train_images/out_%d.png' %
                           (i+ckpt_offset), out[:25].reshape(
                               5, 5, 128, 128).swapaxes(
                               1, 2).reshape(5*128, -1))

                    print("1000 iterations done in %fs" % (time.time()-t))
                    t = time.time()

            saver.save(sess, self.LOG_DIR+"/model.ckpt", self.niters+1)

        return None

    def eval(self, batch, test_name):
        """batch should have the (true_imgs, measurements) scaled to [0,1]"""

        saver = tf.train.Saver()
        ntest = len(batch[0])

        mask = np.load('mask.npy').reshape(128, 128, 1)

        with tf.Session() as sess:
            # initialize before restoring
            sess.run(tf.global_variables_initializer())

            # restore the graph
            saver.restore(sess, tf.train.latest_checkpoint(self.LOG_DIR))
            print('Restore successful')

            loss, out = sess.run([self.loss, self.out], feed_dict={
                self.TRAIN_FLAG: False,
                self.true_img: batch_imgs,
                self.measurements: batch_measurements
            })

            print('Loss on test samples = %f' % loss)

            imsave(self.LOG_DIR+'/test_images/%s_orig_test.png' % test_name,
                   batch_imgs[:12].reshape(
                       1, 12, 128, 128).swapaxes(1, 2).reshape(1*128, -1))

            imsave(self.LOG_DIR+'/test_images/%s_out_test.png' % test_name,
                   out[:12].reshape(
                       1, 12, 128, 128).swapaxes(1, 2).reshape(1*128, -1))

        return None


def main():

    args = parserMethod()

    net = DNN(img_size=128,
              img_ch=1,
              niters=args.niter,
              data_npy=args.data_array,
              meas_npy=args.measurement_array,
              lr=args.learning_rate,
              nbatch=args.batch_size,
              niters=args.niter,
              LOG_DIR='results/'+args.name)

    if args.train:
        if args.resume_from == 0:
            net.train()
        else:
            net.train(rerun=True, ckpt_offset=args.resume_from)

    """Evaluation code. Comment if not required"""
    test_images = np.load('random_samples.npy').astype(
        'float32').reshape(-1, 128, 128, 1)
    measurements = np.load('random_samples_infdb.npy').astype(
        'float32').reshape(-1, 128, 128, 1)
    net.eval(test_images, measurements, 'direct_tr10db_trandom')

    return None


if __name__ == "__main__":
    main()
