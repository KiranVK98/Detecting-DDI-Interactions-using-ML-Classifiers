import numpy as np
import tensorflow as tf

LINEAR_DIR = "dataset/TensorModels/Linear"
DNN_DIR = "dataset/TensorModels/DNN"
DNN_CLASSIFIER_DIR = "dataset/TensorModels/DNN Classifier"


PRICE_NORM_FACTOR = 1
STEPS = 1000

FEATURE_COLUMN = [
    tf.feature_column.numeric_column(key="drug_1", dtype=tf.float32),
    tf.feature_column.numeric_column(key="drug_2", dtype=tf.float32),
    tf.feature_column.numeric_column(key="event", dtype=tf.float32),
    tf.feature_column.numeric_column(key="prr", dtype=tf.float32),
    tf.feature_column.numeric_column(key="drug1_prr", dtype=tf.float32),
    tf.feature_column.numeric_column(key="drug2_prr", dtype=tf.float32),
]



INPUT_DICT = {
    "drug_1": np.array([13.00, 20.00]),
    "drug_2": np.array([20.00, 13.00]),
    "event": np.array([1.00, 2.00]),
    "prr": np.array([0.54, 0.54]),
    "drug1_prr": np.array([12.5, 12.5]),
    "drug2_prr": np.array([12.5, 12.5]),
}

def serving_input_receiver_fn():
    """An input receiver that expects a serialized tf.Example."""
    feature_spec = {
                    "drug_1": tf.FixedLenFeature([0], tf.float32),
                    "drug_2": tf.FixedLenFeature([0], tf.float32),
                    "event": tf.FixedLenFeature([0], tf.float32),
                    "prr": tf.FixedLenFeature([0.0], tf.float32),
                    "drug1_prr": tf.FixedLenFeature([0.0], tf.float32),
                    "drug2_prr": tf.FixedLenFeature([0.0], tf.float32),}

    serialized_tf_example = tf.placeholder(dtype=tf.string,
                                            name='input_example_tensor')
    receiver_tensors = {'examples': serialized_tf_example}
    features = tf.parse_example(serialized_tf_example, feature_spec)
    return tf.estimator.export.ServingInputReceiver(features, receiver_tensors)

