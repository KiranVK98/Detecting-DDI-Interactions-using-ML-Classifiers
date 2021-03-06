"""A dataset loader for imports85.data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import numpy as np
import tensorflow as tf

try:
  import pandas as pd
except ImportError:
  pass


# Order is important for the csv-readers, so we use an OrderedDict here.

defaults = collections.OrderedDict([
    ("drug_1", [0.00]),
    ("drug_2", [0.00]),
    ("event", [0.00]),
    ("prr", [0.0]),
    ("drug1_prr", [0.0]),
    ("drug2_prr", [0.0]),
    ("predicted_value", [0.0]),
])  # pyformat: disable


types = collections.OrderedDict((key, type(value[0]))
                                for key, value in defaults.items())


def _get_imports85():
  path = "../export_data/trainingset2.csv"
#   path = "export_data/trainingset3.csv"
  return path


def dataset(y_name="predicted_value", train_fraction=0.7):
  """Load the imports85 data as a (train,test) pair of `Dataset`.

  Each dataset generates (features_dict, label) pairs.

  Args:
    y_name: The name of the column to use as the label.
    train_fraction: A float, the fraction of data to use for training. The
        remainder will be used for evaluation.
  Returns:
    A (train,test) pair of `Datasets`
  """
  # Load and cache the data
  path = _get_imports85()

  # Define how the lines of the file should be parsed
  def decode_line(line):
    """Convert a csv line into a (features_dict,label) pair."""
    # Decode the line to a tuple of items based on the types of
    # csv_header.values().
    items = tf.decode_csv(line, list(defaults.values()))

    # Convert the keys and items to a dict.
    pairs = zip(defaults.keys(), items)
    features_dict = dict(pairs)

    # Remove the label from the features_dict
    label = features_dict.pop(y_name)

    return features_dict, label

  def has_no_question_marks(line):
    """Returns True if the line of text has no question marks."""
    # split the line into an array of characters
    chars = tf.string_split(line[tf.newaxis], "").values
    # for each character check if it is a question mark
    is_question = tf.equal(chars, "?")
    any_question = tf.reduce_any(is_question)
    no_question = ~any_question

    return no_question

  def in_training_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # If you randomly split the dataset you won't get the same split in both
    # sessions if you stop and restart training later. Also a simple
    # random split won't work with a dataset that's too big to `.cache()` as
    # we are doing here.
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    # Use the hash bucket id as a random number that's deterministic per example
    return bucket_id < int(train_fraction * num_buckets)

  def in_test_set(line):
    """Returns a boolean tensor, true if the line is in the training set."""
    # Items not in the training set are in the test set.
    # This line must use `~` instead of `not` because `not` only works on python
    # booleans but we are dealing with symbolic tensors.
    return ~in_training_set(line)

  base_dataset = (
      tf.data
      # Get the lines from the file.
      .TextLineDataset(path)
      # drop lines with question marks.
      .filter(has_no_question_marks))

  train = (base_dataset
           # Take only the training-set lines.
           .filter(in_training_set)
           # Decode each line into a (features_dict, label) pair.
           .map(decode_line)
           # Cache data so you only decode the file once.
           .cache())

  # Do the same for the test-set.
  test = (base_dataset.filter(in_test_set).cache().map(decode_line))

  return train, test


def raw_dataframe():
  """Load the imports85 data as a pd.DataFrame."""
  # Download and cache the data
  path = _get_imports85()

  # Load it into a pandas dataframe
  df = pd.read_csv(path, names=types.keys(), dtype=types, na_values="?")
  return df


def load_data(y_name="predicted_value", train_fraction=0.7, seed=None):
  """Get the imports85 data set.

  Args:
    y_name: the column to return as the label.
    train_fraction: the fraction of the dataset to use for training.
    seed: The random seed to use when shuffling the data. `None` generates a
      unique shuffle every run.
  Returns:
    a pair of pairs where the first pair is the training data, and the second
    is the test data:
    `(x_train, y_train), (x_test, y_test) = get_imports85_dataset(...)`
    `x` contains a pandas DataFrame of features, while `y` contains the label
    array.
  """
  # Load the raw data columns.
  data = raw_dataframe()

  # Delete rows with unknowns
  data = data.dropna()

  # Shuffle the data
  np.random.seed(seed)

  # Split the data into train/test subsets.
  x_train = data.sample(frac=train_fraction, random_state=seed)
  x_test = data.drop(x_train.index)

  # Extract the label from the features dataframe.
  y_train = x_train.pop(y_name)
  y_test = x_test.pop(y_name)

  return (x_train, y_train), (x_test, y_test)
