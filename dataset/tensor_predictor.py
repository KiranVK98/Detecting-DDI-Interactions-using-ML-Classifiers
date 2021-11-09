import numpy as np
import tensorflow as tf
try:
    import tensor_data as td
except ImportError:
    from .import tensor_data as td


class CreateInput:
    def __init__(self, features):
        print(features)
        self.drug_1 = features.get("drug_1")
        self.drug_2 = features.get("drug_2")
        self.event = features.get("event")
        self.prr = features.get("prr")
        self.drug1_prr = features.get("drug1_prr")
        self.drug2_prr = features.get("drug2_prr")
    
    def get_input_fn(self):
        rdict = {
                "drug_1": np.array([int(self.drug_1),]),
                "drug_2": np.array([int(self.drug_2),]),
                "event": np.array([int(self.event),]),
                "prr": np.array([float(self.prr),]),
                "drug1_prr": np.array([float(self.drug1_prr),]),
                "drug2_prr": np.array([float(self.drug2_prr),])}
        return rdict


def get_linear(features):
    model = tf.estimator.LinearRegressor(
    feature_columns=td.FEATURE_COLUMN,
    model_dir=td.LINEAR_DIR,
    warm_start_from=td.LINEAR_DIR)

    predict_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        features, shuffle=False)
    predict_results = model.predict(input_fn=predict_input_fn)

    cip_predicted = []
    for i, prediction in enumerate(predict_results):
        cip_predicted.append(prediction["predictions"][0])
    return cip_predicted

def get_dnn(features):
    model = tf.estimator.DNNRegressor(
    feature_columns=td.FEATURE_COLUMN,
    hidden_units=[10, 10, 10, 10, 10],
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        l1_regularization_strength=0.001,
        model_dir=td.DNN_DIR,
        warm_start_from=td.DNN_DIR))

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        features, shuffle=False)
    predict_results = model.predict(input_fn=predict_input_fn)

    cip_predicted = []
    for i, prediction in enumerate(predict_results):
        cip_predicted.append(prediction["predictions"][0])
    return cip_predicted

def get_dnn_classifier(features):
    model = tf.estimator.DNNClassifier(
        feature_columns=td.FEATURE_COLUMN,
        hidden_units=[10, 10, 10],
        n_classes=3,
        model_dir=td.DNN_CLASSIFIER_DIR)

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        features, shuffle=False)
    predict_results = model.predict(input_fn=predict_input_fn)
    cip_predicted = []
    for i, prediction in enumerate(predict_results):
        cip_predicted.append(prediction["class_ids"][0])
    return cip_predicted

if __name__ == '__main__':
    cip_linear = get_linear(td.INPUT_DICT)
    # cip_dnn = get_dnn(td.INPUT_DICT)
    # cip_dnn_classifer = get_dnn_classifier(td.INPUT_DICT)
    print("LINEAR: ", cip_linear)
    # print("DNN: ", cip_dnn)
    # print("DNN CLASSIFIER: ", cip_dnn_classifer)
    
