#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import math
import os

import os

from os.path import join
from os.path import isdir
from os import listdir
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
from sklearn import metrics
import yaml

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile)

# Parameters
# ==================================================

# Data Parameters

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "./runs/1493013586/checkpoints", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", True, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

datasets = None

# CHANGE THIS: Load data. Load your own data here
dataset_name = cfg["datasets"]["default"]
if FLAGS.eval_train:
    if dataset_name == "mrpolarity":
        datasets = data_helpers.get_datasets_mrpolarity(cfg["datasets"][dataset_name]["positive_data_file"]["path"],
                                             cfg["datasets"][dataset_name]["negative_data_file"]["path"])
    elif dataset_name == "20newsgroup":
        datasets = data_helpers.get_datasets_20newsgroup(subset="test",
                                              categories=cfg["datasets"][dataset_name]["categories"],
                                              shuffle=cfg["datasets"][dataset_name]["shuffle"],
                                              random_state=cfg["datasets"][dataset_name]["random_state"])
    elif dataset_name == "localdata":
        datasets = data_helpers.get_datasets_localdata(container_path=cfg["datasets"][dataset_name]["test_path"],
                                                       categories=None,
                                                       shuffle=False)
    x_raw, y_test = data_helpers.load_data_labels(datasets)
    y_test = None
else:
    if dataset_name == "mrpolarity":
        x_raw = ["a masterpiece four years in the making", "everything is off."]
        y_test = [1, 0]
    else:
        x_raw = ["The number of reported cases of gonorrhea in Colorado increased",
                 "I am in the market for a 24-bit graphics card for a PC"]
        y_test = [2, 1]

# get meta info
container_path = cfg["datasets"][dataset_name]["container_path"]
folders = [f for f in sorted(listdir(container_path))
           if isdir(join(container_path, f))]
targets = dict()
for label, folder in enumerate(folders):
    targets[label] = folder
filenames = datasets['filenames']

# get split  list
step = 1000.0
loop_num = int(math.ceil(len(x_raw) / step))
print(len(x_raw))
print(len(datasets['filenames']))
print(loop_num)
raw_list = []
ip_list = []

start = 0
for i in range(loop_num):
    end = start + int(step)
    print("%d->%d" %(start, end, ))

    raw_list.append(x_raw[start:end])
    ip_list.append([os.path.basename(filename) for filename in filenames[start:end]])

    start = end

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

# Collect the predictions here
all_predictions = []

print("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        index = 1
        for x_raw_one_batch in raw_list:
            x_test = np.array(list(vocab_processor.transform(x_raw_one_batch)))
            one_raw_predictions = []

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                one_raw_predictions = np.concatenate([one_raw_predictions, batch_predictions])

            all_predictions.append(one_raw_predictions)
            print("predict done : %d" % (index, ))
            index += 1

for i in range(loop_num):
    one_batch_predictions = all_predictions[i]
    x_raw_batch = raw_list[i]
    ips = ip_list[i]
    cats = []
    for p in one_batch_predictions:
        cat_key = int(p)
        if cat_key in targets:
            cats.append(targets[cat_key])
        else:
            cats.append('unknown')

    # Save the evaluation to a csv
    predictions_human_readable = np.column_stack((ips, np.array(x_raw_batch), one_batch_predictions, cats))
    out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction_%d.csv" % (i, ))
    print("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
