"""Linear regression using the LinearRegressor Estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import imports85
import tensor_data as td

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)


def train_linear():
  """Builds, trains, and evaluates the model."""
  (train, test) = imports85.dataset()

  # Switch the labels to units of thousands for better convergence.
  def to_thousands(features, labels):
    return features, labels / td.PRICE_NORM_FACTOR

  train = train.map(to_thousands)
  test = test.map(to_thousands)

  def input_train():
    return (
        train.shuffle(1000).batch(128)
        .repeat().make_one_shot_iterator().get_next())

  def input_test():
    return (test.shuffle(1000).batch(128)
            .make_one_shot_iterator().get_next())

  sess = tf.Session()
  # Build the Estimator.
  # model = tf.estimator.BoostedTreesRegressor(feature_columns=feature_columns, n_batches_per_layer=32)
  model = tf.estimator.LinearRegressor(
    feature_columns=td.FEATURE_COLUMN,
    model_dir=td.LINEAR_DIR)

  #Train the model.
  #By default, the Estimators log output every 100 steps.
  model.train(input_fn=input_train, steps=td.STEPS)

  # Evaluate how the model performs on data it has not yet seen.
  eval_result = model.evaluate(input_fn=input_test)

  # The evaluation returns a Python dictionary. The "average_loss" key holds the
  # Mean Squared Error (MSE).
  average_loss = eval_result["average_loss"]

  model.export_savedmodel(
    td.LINEAR_DIR,
    td.serving_input_receiver_fn,
    strip_default_attrs=False)

def train_dnn():
  """Builds, trains, and evaluates the model."""
  (train, test) = imports85.dataset()

  def to_thousands(features, labels):
    return features, labels / td.PRICE_NORM_FACTOR

  train = train.map(to_thousands)
  test = test.map(to_thousands)

  def input_train():
    return (
        train.shuffle(1000).batch(128)
        .repeat().make_one_shot_iterator().get_next())

  def input_test():
    return (test.shuffle(1000).batch(128)
            .make_one_shot_iterator().get_next())

  model = tf.estimator.DNNRegressor(
    feature_columns=td.FEATURE_COLUMN,
    hidden_units=[10, 10, 10, 10, 10],
    model_dir=td.DNN_DIR)

  model.train(input_fn=input_train, steps=td.STEPS)

  eval_result = model.evaluate(input_fn=input_test)

  average_loss = eval_result["average_loss"]
  print(average_loss)

  model.export_savedmodel(
    td.DNN_DIR,
    td.serving_input_receiver_fn,
    strip_default_attrs=False)


def train_dnn_classifier():
  """Builds, trains, and evaluates the model."""
  (train, test) = imports85.dataset()
  def to_thousands(features, labels):
    return features, labels

  train = train.map(to_thousands)
  test = test.map(to_thousands)

  def input_train():
    return (
        train.shuffle(1000).batch(128)
        .repeat().make_one_shot_iterator().get_next())

  def input_test():
    return (test.shuffle(1000).batch(128)
            .make_one_shot_iterator().get_next())

  model = tf.estimator.DNNClassifier(
    feature_columns=td.FEATURE_COLUMN,
    hidden_units=[10, 10, 10],
    n_classes=3,
    model_dir=td.DNN_CLASSIFIER_DIR)

  model.train(input_fn=input_train, steps=td.STEPS)

  eval_result = model.evaluate(input_fn=input_test)

  average_loss = eval_result["average_loss"]
  print(average_loss)

  model.export_savedmodel(
    td.DNN_CLASSIFIER_DIR,
    td.serving_input_receiver_fn,
    strip_default_attrs=False)


if __name__ == "__main__":
  # The Estimator periodically generates "INFO" logs; make these logs visible.
  tf.logging.set_verbosity(tf.logging.INFO)
  # tf.app.run(main=main)
  train_linear()
  # train_dnn()
  # train_dnn_classifier()
