from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time
import os
from six.moves import cPickle
from subprocess import call

from utils import TextLoader
from model import Model

from six import text_type

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on bar lines')
    parser.add_argument('-T', type=str, default="A Tune",
        help="Name of the tune")
    parser.add_argument('-R', type=str, default="reel",
        help="Type of the tune (reel, air, jig, polka, etc.)")
    parser.add_argument('-M', type=str, default="4/4",
        help="Time signature")
    parser.add_argument('-L', type=str, default="1/8",
        help="Length of beat (This should nearly always be 1/8)")
    parser.add_argument('-K', type=str, default="Dmaj",
        help="Key (Gmaj and Dmaj work best)")
    parser.add_argument('--tune_dir', type=str, default=".",
        help='Where to save the svg of the generated tune')

    args = parser.parse_args()
    tune = sample(args)
    with open(os.path.join(args.tune_dir + "/out.abc"), "w") as text_file:
        text_file.write(tune)
        os.chdir(args.tune_dir)
        print(os.getcwd())
        call("pwd")
        call("abcm2ps -g -O = out.abc");
		
		 
def sample(args):
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, True)
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            tune = (model.sample_tune(
                sess, chars, vocab, args.T, args.R,
                int(args.M[0]), int(args.M[2]),
                int(args.L[0]), int(args.L[2]),
                args.K, args.sample))
            tune = tune[:-1]
            print(tune)
            return tune
if __name__ == '__main__':
    main()

