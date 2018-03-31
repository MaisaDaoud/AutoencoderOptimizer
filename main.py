import csv
import os

#import hdfs3 as hdfs3
import numpy as np
import tensorflow as tf
import pprint
from Autoencoder import Autoencoder
from DataLoader import DataLoader
import pandas as pd

flags = tf.app.flags
flags.DEFINE_integer("epochs", 10, "number of training epochs")
flags.DEFINE_float("learning_rate", 0.0001, "learning rate")
flags.DEFINE_integer("batch_size", 25, "number of samples/batch")
flags.DEFINE_string("checkpoint_dir","checkpoint","checkpoint drectory")
flags.DEFINE_string("dataset", "THCA", "dataset file name")
flags.DEFINE_boolean("train", True, "true to train , false to test the model")
flags.DEFINE_boolean("generate",False,"true to generate synthetic minority samples")
flags.DEFINE_list("n_layers", [1000,50,20], "# of neurons /layer in the encoder side")
flags.DEFINE_integer("class_index", -1, "class index")

FLAGS = flags.FLAGS


def main(_):
    # ... printing the flags ....
    pp=pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)


    # define and load the data
    dataset_name = FLAGS.dataset
    class_index=FLAGS.class_index
    data_loader=DataLoader(dataset_name,class_index,FLAGS.train)


    # start training
    train_data(dataset_name,data_loader.data_train, data_loader.data_test, 1, data_loader.cancer_training_size, data_loader.cancer_testing_size) #1: only one run
    print("[****]Done Run :", 1)



def train_data (dataset_name,training_data, testing_data, run, cancer_training_size, cancer_testing_size):

    # with tf.Session() as sess:
    sess = tf.Session()

    auto = Autoencoder(
        sess,
        run=run,
        cancer_training_size=cancer_training_size,
        cacner_testing_size=cancer_testing_size,
        dataset_name=dataset_name,
        epochs=FLAGS.epochs,
        learning_rate=FLAGS.learning_rate,
        batch_size=FLAGS.batch_size,
        n_layers=FLAGS.n_layers,
        training_data=training_data,
        testing_data=testing_data,
        checkpoint_dir=FLAGS.checkpoint_dir,
        train=FLAGS.train,
        generate=FLAGS.generate)
    sess.close()


if __name__ == "__main__":
    tf.app.run()
