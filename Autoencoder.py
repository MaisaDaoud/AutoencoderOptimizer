import csv
import numpy as np
import os
import tensorflow as tf
#from Layer import Layer
import time
from six.moves import xrange
from pyswarm1 import pso


class Autoencoder(object):

    def __init__(self, sess, epochs=1, run=1, learning_rate=0.001, batch_size=100, n_layers=2,
                 checkpoint_dir="checkpoint",
                 training_data="data_train", cancer_training_size=1, cacner_testing_size=1, testing_data="data_test",
                 train=True, dataset_name="READ", generate=False):

        self.batch_size = batch_size
        self.checkpoint_dir = os.path.join(checkpoint_dir, dataset_name)
        self.epochs = epochs
        self.dataset_name = dataset_name
        self.learning_rate = learning_rate
        self.n_layers = [int(float(n_layers[x])) for x in range (0,len(n_layers))]
        print("self.n_layers", self.n_layers)
        self.run = run
        self.sess = sess
        self.testing_data = testing_data
        self.training_data = training_data
        self.input_length = len(self.testing_data[1])
        self.output_length = self.input_length
        self.cancer_testing_size = cacner_testing_size
        self.train = train
        if self.train:
            self.training_size = len(self.training_data)  # training_size
            print("self.trainning size=", self.training_size)
            self.cancer_training_size = cancer_training_size
        self.generate = generate
        # define the layers

        self.weights, self.baiases = self.initialize_layers(self.n_layers, self.input_length, self.output_length)
        '''
        self.layers= self.initialize_layers(self.n_layers,self.input_length,self.output_length)
   
        for i in (0,len(self.layers)):
         print("weights = ",self.layers[i].weights, "bias = ",self.layers[i].bias)
        '''

        self.build_model()

    def initialize_layers(self, n_layers, input_length, output_length):

        weights = {}
        biases = {}
        # encoder weights and biases
        weights['encoder_h1'] = tf.Variable(tf.random_normal([input_length, n_layers[0]]))
        biases['encoder_b1'] = tf.Variable(tf.random_normal([n_layers[0]]))

        print("[*] The Model has: Input layer,", len(n_layers) * 2, "hidden layers ", "and Output layer")
        for l in range(0, len(n_layers) - 1):
            weights['encoder_h' + str(l + 2)] = tf.Variable(tf.random_normal([n_layers[l], n_layers[l + 1]]))
            biases['encoder_b' + str(l + 2)] = tf.Variable(tf.random_normal([self.n_layers[l + 1]]))

        # decoder weights and biases
        for l in range(0, len(n_layers) - 1):
            weights['decoder_h' + str(l + 1)] = tf.Variable(
                tf.random_normal([n_layers[-1 * l - 1], n_layers[-1 * l - 2]]))
            biases['decoder_b' + str(l + 1)] = tf.Variable(tf.random_normal([self.n_layers[-1 * l - 2]]))
        weights['decoder_h' + str(len(n_layers))] = tf.Variable(
            tf.random_normal([n_layers[0], output_length]))
        biases['decoder_b' + str(len(n_layers))] = tf.Variable(tf.random_normal([output_length]))

        return weights, biases

    '''

    def initialize_layers(self, n_layers, input_length, output_length):
        layers=[]
        layers.append(Layer('encoder_h1',input_length,n_layers[0]))
        print("[*] The Model has: Input layer,", len(n_layers) * 2, "hidden layers ", "and Output layer")
        for l in range(0, len(n_layers) - 1):
            layers.append(Layer('encoder_h' + str(l + 2),n_layers[l], n_layers[l + 1]))
        for l in range(0, len(n_layers) - 1):
            layers.append(Layer('decoder_h' + str(l + 2), n_layers[l], n_layers[l + 1]))
        layers.append(Layer('decoder_h' + str(len(n_layers)),n_layers[0], output_length))

        return layers

    '''

    def build_model(self):

        self.saver = tf.train.Saver()
        if (self.train):

            self.training()  # training_samples)
        else:
            self.test(self.testing_data, "testing_reps", self.cancer_testing_size)
        if (self.generate):
            self.PSO_optimizer(self.testing_data, self.cancer_testing_size, self.sess)

    def training(self):
        X = tf.placeholder('float', [None, self.input_length])
        self.e = self.encoder(X)
        self.d = self.decoder(self.e)

        # TODO implement other cost and optimizers
        cost = tf.reduce_mean(tf.pow(self.d - X, 2))
        optimizer1 = tf.train.AdamOptimizer(self.learning_rate).minimize(cost)
        optimizer2 = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(cost)
        # Initializing the variables
        try:
            tf.global_variables_initializer().run(session=self.sess)
        except:
            tf.initialize_all_variables().run(session=self.sess)
        counter = 1
        display_step = 2
        # start_time = time.time()

        could_load = self.load(self.checkpoint_dir, "training")

        if could_load:

            print(" [*] Load SUCCESS")
        else:

            print(" [!] Load failed...")

        for epoch in xrange(self.epochs):

            # number of patches
            if self.training_size < self.batch_size:
                self.batch_size = self.training_size
                batch_idxs = 1
            else:
                batch_idxs = min(len(self.training_data), self.training_size) // self.batch_size

            # print("[***] len(self.data) = ",len(self.data))
            # start trainning
            for idx in xrange(batch_idxs):
                batch_samples = self.training_data[
                                idx * self.batch_size:(idx + 1) * self.batch_size]  # np.array(batch).astype(np.float32)
                '''
                if epoch / 2 == 0:
                    optimizer = optimizer2
                # print("[***] patch samples shape = ",batch_samples.shape)
                
                else :
                    optimizer=optimizer1
                '''
                optimizer = optimizer1
                _, c = self.sess.run([optimizer, cost], feed_dict={
                    X: self.mask_noise(np.reshape(batch_samples, [self.batch_size, self.input_length]), 20)})

            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c))

            if np.mod(counter, 2) == 0:
                self.save(self.checkpoint_dir, counter)
            counter += 1
        #Generate training reps
        self.test(self.training_data, "data_train", self.cancer_training_size)
        #Generate testing  reps
        self.test(self.testing_data, "data_test", self.cancer_testing_size)
        if self.generate:
            self.PSO_optimizer(self.training_data, self.cancer_training_size, self.sess)


    def encoder(self, x):
        layer = x

        for l in xrange(0, len(self.n_layers)):
            name_w = "encoder_h" + str(l + 1)
            name_b = "encoder_b" + str(l + 1)
            # TODO change the activation function
            layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, self.weights[name_w]), self.baiases[name_b]))

        return layer

    def decoder(self, x):
        layer = x
        for l in xrange(0, len(self.n_layers)):
            name_w = "decoder_h" + str(l + 1)
            name_b = "decoder_b" + str(l + 1)
            # TODO change the activation function
            layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, self.weights[name_w]), self.baiases[name_b]))
        return layer

    def test(self, data, name, cancer_training_size):
        X = tf.placeholder('float', [None, self.input_length])
        self.e = self.encoder(X)
        self.d = self.decoder(self.e)

        cost = tf.reduce_mean(tf.pow(self.d - X, 2), 1)
        # summary_z =  tf.summary.scalar(cost.op.name, tf.squeeze(cost))
        summary_merge = tf.summary.merge_all()
        # on each desired step save:

        could_load = self.load(self.checkpoint_dir, "testing")
        if could_load:
            print(" [*] Load SUCCESS")
            reps_dir = "Representations/" + self.dataset_name + "/Autoencoder"
            file_dir = reps_dir
            if not os.path.exists(file_dir):
                os.makedirs(reps_dir)

            reps, c = self.sess.run([self.e, cost], feed_dict={X: np.reshape(data, [len(data), self.input_length])})

            # attach header
            header = range(self.n_layers[-1])
            reps = np.vstack((header, reps))
            # attach class

            class_att = [[0 for j in range(1)] for i in range(len(reps))]
            for idx in xrange(len(reps)):
                if idx < (cancer_training_size + 1):
                    class_att[idx][0] = 1
                else:
                    class_att[idx][0] = 0

            reps = np.hstack((reps, class_att))
            np.savetxt(os.path.join(file_dir, "{}_{}".format(name, "reps.csv")), np.transpose(reps, (0, 1)),
                       delimiter=",")
        else:
            print(" [!] Load failed...")

    def PSO_optimizer(self, data_samples, cancer_size, sess):
        # load the model
        could_load = self.load(self.checkpoint_dir,"generating new data")
        if could_load:

            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        rep_length = self.n_layers[-1]  # self.n_hidden_2#
        # ................. Optimization ..........................................

        lb = 0 * (np.ones((rep_length)).astype(np.float64))  # [10, 1, 0.01]
        ub = (np.ones((rep_length)).astype(np.float64))  # [10, 1, 0.01]
        normal_size = len(data_samples) - cancer_size
        replication = np.int(
            cancer_size / normal_size + 1) * normal_size  # replication=(np.int(cancer_size/normal_size)+1)*normal_size)
        collection = np.zeros([replication, rep_length])  # self.n_input])#np.empty([128, 784])
        collection2 = np.zeros([replication, rep_length])  # self.n_input])#np.empty([128, 784])
        samplesLoss2 = np.zeros([replication])
        optimized_generatedSamples = np.zeros([replication, self.input_length])  # self.n_input])#np.empty([128, 784])

        samples_cost = np.zeros(replication)
        X = tf.placeholder('float', [None, self.n_layers[-1]])  # [None,self.n_hidden_2]) #[None,self.n_hidden_4])
        opt_decoder = self.decoder(X)
        # Create a file containign the generated_samples
        # encoder part
        weights = {}
        baiases = {}
        for l in range(0, len(self.n_layers)):
            weights['encoder_h' + str(l + 1)] = self.weights['encoder_h' + str(l + 1)].eval(sess)
            baiases['encoder_b' + str(l + 1)] = self.baiases['encoder_b' + str(l + 1)].eval(sess)
            weights['decoder_h' + str(l + 1)] = self.weights['decoder_h' + str(l + 1)].eval(sess)
            baiases['decoder_b' + str(l + 1)] = self.baiases['decoder_b' + str(l + 1)].eval(sess)
        f = 0
        for idx1 in range(cancer_size // normal_size + 1):
            for idx in range(normal_size):
                print("round ", idx1, "samples", idx)
                global index
                index = idx
                original_sample = data_samples[idx + cancer_size]
                # print("[*]index = .......................",index)
                args = (original_sample, rep_length, weights, baiases)
                xopt, fopt = pso(self.fitness, lb, ub, swarmsize=4, maxiter=1, args=args)  # 300 , 2
                optimized_code = xopt.reshape((1, rep_length))
                optimized_reps = optimized_code  # self.sess.run(y_opt, feed_dict={optimized_x:optimized_code }) #batch_size
                samplesLoss2[f] = fopt
                collection[f] = optimized_reps[0]  # np.append(collection,optimized_code,axis=0)
                f = f + 1
        # redecode

        # self.save_train_files(collection, self.run, "Optimized", "optimized_traning_reps.csv", self.dataset_name)
        reps_dir = "Representations/" + self.dataset_name + "/Optimized"
        file_dir = reps_dir
        if not os.path.exists(file_dir):
            os.makedirs(reps_dir)
        np.savetxt(os.path.join(file_dir, "{}_{}".format("optimized_", "reps.csv")), collection, delimiter=",")

    def fitness(self, x, *args):

        original_sample, rep_length, weights, biases = args
        l = x
        idx = 1
        for i in range(len(weights) // 2, len(weights)):
            l = 1 / (1 + np.exp(-1 * (np.add(np.dot(l, weights['decoder_h' + str(idx)]),
                                             biases['decoder_b' + str(idx)]))))
            idx = idx + 1
        d_loss = np.sum(np.power((l - original_sample), 2)) / len(original_sample)

        return d_loss

    def xavier_init(self, nin, nout, const=1):
        low = -const * np.sqrt(1 / (nin + nout))
        high = const * np.sqrt(1 / (nin + nout))

        return tf.random_uniform((nin, nout), minval=low, maxval=high)

    # Noising Method
    def mask_noise(self, x, v):
        x_noise = x.copy()

        n_samples = x.shape[0]
        n_features = x.shape[1]

        for i in range(n_samples):
            mask = np.random.randint(0, n_features, v)

            for m in mask:
                x_noise[i][m] = 0.

        return x_noise

    @property
    def model_dir(self):

        return "{}_{}_{}_{}_{}".format(
            "dataset_train", self.batch_size, self.input_length, self.output_length, self.run)

    def save(self, checkpoint_dir, step):
        model_name = "Autoencoder.model"
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name))

    def load(self, checkpoint_dir, phase):
        import re

        print("[*] Reading checkpoints for", phase)
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)
        print("[*] Loading previously trained model from", checkpoint_dir)
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

        if ckpt and ckpt.model_checkpoint_path:
            try:
                ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

                self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))

                # print(" [*] Success to read {}".format(ckpt_name))
                return True
            except:
                print("[!] Deleting the pre-trained model, it has different config.")

        else:
            print(" [*] Failed to find a checkpoint")
            return False
