import os
import glob
import time
import numpy as np
from model import nn
import tensorflow as tf
from pickle import load
import utils.inputpipe as ip
import matplotlib.pyplot as plt
from parser import parserMethod
from scipy.misc import imsave, imread
import tensorflow.contrib.layers as tcl


def make_if_not_exist(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def summary(name, tensor):
    tf.summary.scalar(name, tensor)
    return tensor


class SubNet():
    """
    Implements the CNN for tomographic reconstruction
    on a coarse mesh. Different meshes would be different
    instantiations of the same class.

    """

    def __init__(self, **kwargs):

        self.lr = kwargs['lr']
        self.ntri = kwargs['ntri']
        self.nproj = kwargs['nproj']
        self.nbatch = kwargs['nbatch']
        self.ntrain = kwargs['ntrain']
        self.img_ch = kwargs['img_ch']
        self.niters = kwargs['niters']
        self.LOG_DIR = kwargs['LOG_DIR']
        self.data_npy = kwargs['data_npy']
        self.meas_npy = kwargs['meas_npy']
        self.PROJ_DIR = kwargs['PROJ_DIR']
        self.img_size = kwargs['img_size']

        make_if_not_exist(self.LOG_DIR)
        make_if_not_exist(self.LOG_DIR+'/train_images')
        make_if_not_exist(self.LOG_DIR+'/test_images')

        self.proj, self.proj_inv = self.load_projectors(
            self.nproj, self.PROJ_DIR)

        # reset default graph
        tf.reset_default_graph()

        # placeholders:
        # required_projection output
        self.true_img = tf.placeholder(tf.float32,
                                       [None, self.img_size,
                                        self.img_size, self.img_ch])

        self.measurements = tf.placeholder(tf.float32,
                                           [None, self.img_size,
                                            self.img_size, self.img_ch])

        # training flag
        self.TRAIN_FLAG = tf.placeholder(tf.bool)

        # projector
        self.P = tf.placeholder(
            tf.float32, [None, self.ntri, self.img_size**2])

        # projector pseudoinverse
        self.Pinv = tf.placeholder(
            tf.float32, [None, self.img_size**2, self.ntri])

        # build the net
        self.build_model()

        return

    def load_projectors(self, n, dir):
        P = np.zeros((n, self.ntri, self.img_size**2))
        Pinv = np.zeros((n, self.img_size**2, self.ntri))

        print('Reading in %d projectors...' % n)

        for i in range(n):
            with open(dir+'/P%d.pkl' % i, 'rb') as f:
                P[i] = load(f)
            with open(dir+'/Pinv%d.pkl' % i, 'rb') as f:
                Pinv[i] = load(f)

        print('Projectors read in!')

        return P, Pinv

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

    def apply_forward(self, out):
        """Forward op"""

        # out has shape (batch_size, img_size**2).
        # R has shape (batch_size, Ntri, img_size**2)
        # we add an axis=1 to out and multiply with R
        # By broadcast rules, this would give
        # (batch_size, Ntri, img_size**2) and reduce_mean
        # at axis 2, gives the low-dim projection of generated image

        out = tf.expand_dims(out, 1)*self.R
        out = tf.reduce_sum(out, axis=2)

        # Pinv is similar
        out = tf.expand_dims(out, 1)*self.Rinv
        out = tf.reduce_sum(out, axis=2)

        return out

    def apply_projection(self, out):
        """Projection layer"""

        # out has shape (batch_size, img_size**2).
        # P has shape (batch_size, Ntri, img_size**2)
        # we add an axis=1 to out and multiply with P
        # By broadcast rules, this would give
        # (batch_size, Ntri, img_size**2) and reduce_mean
        # at axis 2, gives the low-dim projection of generated image

        out = tf.expand_dims(out, 1)*self.P
        out = tf.reduce_sum(out, axis=2)

        # Pinv is similar
        out = tf.expand_dims(out, 1)*self.Pinv
        out = tf.reduce_sum(out, axis=2)

        return out

    def build_model(self):

        # concat the projection op

        # first we reshape P to get channels last
        # P shape should be (batch_size, img_size, img_size, ntri)
        reshape_P = tf.transpose(tf.reshape(self.P,
                                            (-1, self.ntri, self.img_size,
                                             self.img_size)),
                                 perm=[0, 2, 3, 1], name='reshape_proj')

        # next we concat the projection matrix to the image
        concat = tf.concat((self.measurements, reshape_P),
                           axis=3, name='concat_inp')

        self.out, self.reg_loss = nn(concat,
                                     reuse=False,
                                     TRAIN_FLAG=self.TRAIN_FLAG,
                                     nchstart=64,
                                     act_fn=tf.nn.leaky_relu)

        self.out = self.apply_projection(self.out)

        # get projection of actual image
        flat_out = tcl.flatten(self.true_img)
        flat_out = self.apply_projection(flat_out)

        self.proj_out = tf.reshape(
            flat_out, (-1, self.img_size, self.img_size))

        self.loss = tf.reduce_mean(
            tf.square(self.out - flat_out)) + self.reg_loss

        # update_ops required for batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.NNtrain = tf.train.AdamOptimizer(
                self.lr).minimize(self.loss)

        summary("nn_loss", self.loss)
        self.summary_op = tf.summary.merge_all()
        print('Model built!')

        return None

    def train(self, rerun=False, ckpt_offset=0):
        """Runs the training loop"""
        self.read_data()

        saver = tf.train.Saver()

        t = time.time()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            if rerun:
                saver.restore(sess, tf.train.latest_checkpoint(self.LOG_DIR))

            summary_writer = tf.summary.FileWriter(self.LOG_DIR, sess.graph)

            print('Beginning training @ 1G!')

            for i in range(self.niters):
                if (i+1) % 2000 == 0:
                    """Validation loss loop + CKPT loop"""
                    saver.save(sess, self.LOG_DIR+"/model.ckpt", i+ckpt_offset)

                    idx = np.random.choice(np.arange(self.nsamples_test),
                                           self.nbatch,
                                           replace=False)

                    batch = self.data_test[idx]
                    measurements = self.measurements_test[idx]

                    idx = np.random.choice(
                        np.arange(self.nproj), self.nbatch, replace=False)
                    p = self.proj[idx]
                    pinv = self.proj_inv[idx]

                    feed_dict_in = {self.true_img: batch,
                                    self.TRAIN_FLAG: False,
                                    self.measurements: measurements,
                                    self.Pinv: pinv,
                                    self.P: p}

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

                idx = np.random.choice(
                    np.arange(self.nproj), self.nbatch, replace=False)
                p = self.proj[idx]
                pinv = self.proj_inv[idx]

                feed_dict_in = {self.true_img: batch,
                                self.TRAIN_FLAG: True,
                                self.measurements: measurements,
                                self.Pinv: pinv,
                                self.P: p}

                _ = sess.run(self.NNtrain, feed_dict=feed_dict_in)

                if (i+1) % 1000 == 0:
                    loss, y, out, projected_img = sess.run(
                        [self.loss, self.measurements,
                         self.out, self.proj_out],
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
                    imsave(self.LOG_DIR+'/train_images/projected_img_%d.png' %
                           (i+ckpt_offset), projected_img[:25].reshape(
                               5, 5, 128, 128).swapaxes(
                               1, 2).reshape(5*128, -1))

                    print("1000 iterations done in %fs" % (time.time()-t))
                    t = time.time()

            saver.save(sess, self.LOG_DIR+"/model.ckpt", self.niters+1)

        return None

    def eval(self, batch, test_name):
        """batch should have (true_imgs, measurements)"""

        saver = tf.train.Saver()
        ntest = len(batch[0])

        mask = np.load('mask.npy').reshape(128, 128, 1)

        with tf.Session() as sess:
            # initialize before restoring
            sess.run(tf.global_variables_initializer())

            # restore the graph
            saver.restore(sess, tf.train.latest_checkpoint(self.LOG_DIR))
            print(self.LOG_DIR)

            out = np.zeros((ntest, 350, 16384))

            for samp in range(ntest):
                input_img = np.zeros((50, 128, 128, 1))
                input_measurements = np.zeros((50, 128, 128, 1))

                input_img[:50] = batch[0][samp].reshape(128, 128, 1)
                input_measurements[:50] = batch[1][samp].reshape(128, 128, 1)

                for i in range(7):
                    p = self.proj[i*50:(i+1)*50]
                    pinv = self.proj_inv[i*50:(i+1)*50]

                    loss, out[samp, i*50:(i+1)*50], proj_out = sess.run(
                        [self.loss, self.out, self.proj_out], feed_dict={
                            self.true_img: input_img,
                            self.P: p,
                            self.Pinv: pinv,
                            self.TRAIN_FLAG: False,
                            self.measurements: input_measurements,
                        })

                    print('Loss on test samples = %f' % loss)

            np.save('%s_out%d.npy' % (test_name, 350), out)
            print('Eval run successfully!')

        return None


def main():

    args = parserMethod()

    net = SubNet(img_size=128,
                 img_ch=1,
                 niters=args.niter,
                 data_npy=args.data_array,
                 meas_npy=args.measurement_array,
                 ntrain=args.ntrain,
                 nbatch=args.batch_size,
                 lr=args.learning_rate,
                 ntri=args.dim_rand_subspace,
                 nproj=args.num_projectors_to_use,
                 PROJ_DIR=args.projectors_dir,
                 LOG_DIR='results/'+args.name)

    if args.train:
        if args.resume_from == 0:
            net.train()
        else:
            net.train(rerun=True, ckpt_offset=args.resume_from)

    """Evaluation code. Comment if not required"""
    test_images = np.load('geo_originals.npy').astype(
        'float32').reshape(-1, 128, 128, 1)
    measurements = np.load('geo_pos_recon_10db.npy').astype(
        'float32').reshape(-1, 128, 128, 1)
    net.eval((test_images, measurements), 'recon_tr10db_t10db')

    measurements = np.load('geo_pos_recon_infdb.npy').astype(
        'float32').reshape(-1, 128, 128, 1)
    net.eval((test_images, measurements), 'recon_tr10db_tinfdb')

    return None


if __name__ == "__main__":
    main()
