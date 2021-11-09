import pickle
import numpy as np

import array
import random
from sklearn.preprocessing import MinMaxScaler


from deap import algorithms, base, creator, tools

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, f1_score, precision_score





def fitFunction(individual, parameter1, parameter2):
    real_labels = parameter1
    multiple_prediction = parameter2
    ensemble_prediction = np.zeros(len(real_labels))

    print(len(multiple_prediction))
    for i in range(0, len(multiple_prediction)):
        # ensemble_prediction = ensemble_prediction + individual[i]*multiple_prediction[i]
        np.append(ensemble_prediction, individual[i]*multiple_prediction[i])
    precision, recall, pr_thresholds = precision_recall_curve(
        real_labels, ensemble_prediction)
    aupr_score = auc(recall, precision)
    return (aupr_score),

def getParamter(real_matrix, multiple_matrix, testPosition):

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", array.array, typecode='d',
                   fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    # Attribute generator
    toolbox.register("attr_float", random.uniform, 0, 1)
    # Structure initializers
    variable_num = len(multiple_matrix)
    toolbox.register("individual", tools.initRepeat,
                     creator.Individual, toolbox.attr_float, variable_num)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    #################################################################################################
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    multiple_prediction = []
    for i in range(0, len(multiple_matrix)):
        predicted_probability = []
        predict_matrix = multiple_matrix[i]
        for j in range(0, len(testPosition)):
            predicted_probability.append(
                predict_matrix[testPosition[j][0], testPosition[j][1]])
        normalize = MinMaxScaler()
        predicted_probability = np.array(predicted_probability).reshape(-1, 1)
        predicted_probability = normalize.fit_transform(predicted_probability)
        multiple_prediction.append(predicted_probability)

    #################################################################################################
    print(len(real_labels), len(multiple_prediction))
    # real_labels = real_labels[0:1000]
    toolbox.register("evaluate", fitFunction,
                     parameter1=real_labels, parameter2=multiple_prediction)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    random.seed(0)
    pop = toolbox.population(n=100)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50,
                                   stats=stats, halloffame=hof, verbose=True)
    pop.sort(key=lambda ind: ind.fitness, reverse=True)
    print(pop[0])
    return pop[0]


def ensemble_scoring(real_matrix, multiple_matrix, testPosition, weights, cf1, cf2):
    
    real_labels = []
    for i in range(0, len(testPosition)):
        real_labels.append(real_matrix[testPosition[i][0], testPosition[i][1]])

    multiple_prediction = []
    for i in range(0, len(multiple_matrix)):
        predicted_probability = []
        predict_matrix = multiple_matrix[i]
        for j in range(0, len(testPosition)):
            predicted_probability.append(
                predict_matrix[testPosition[j][0], testPosition[j][1]])
        normalize = MinMaxScaler()
        predicted_probability = np.array(predicted_probability).reshape(-1, 1)
        predicted_probability = normalize.fit_transform(predicted_probability)
        predicted_probability = np.array(predicted_probability)
        multiple_prediction.append(predicted_probability)
    ensemble_prediction = np.zeros(len(real_labels))
    for i in range(0, len(multiple_matrix)):
        # ensemble_prediction = ensemble_prediction + \
        #     weights[i] * multiple_prediction[i]
        numpy.append(ensemble_prediction, weights[i] * multiple_prediction[i])
    ensemble_prediction_cf1 = np.zeros(len(real_labels))
    ensemble_prediction_cf2 = np.zeros(len(real_labels))
    for i in range(0, len(testPosition)):
        vector = []
        for j in range(0, len(multiple_matrix)):
            vector.append(
                multiple_matrix[j][testPosition[i][0], testPosition[i][1]])
        
        vector = np.array(vector).reshape(-1,1)

        aa = cf1.predict_proba(vector)
        print(aa)
        ensemble_prediction_cf1[i] = (cf1.predict_proba(vector))[0][1]
        ensemble_prediction_cf2[i] = (cf2.predict_proba(vector))[0][1]

    normalize = MinMaxScaler()
    ensemble_prediction = normalize.fit_transform(ensemble_prediction)

    result = calculate_metric_score(real_labels, ensemble_prediction)
    result_cf1 = calculate_metric_score(real_labels, ensemble_prediction_cf1)
    result_cf2 = calculate_metric_score(real_labels, ensemble_prediction_cf2)

    return result, result_cf1, result_cf2



if __name__ == "__main__":
    real_matrix = np.load("real_matrix.pkl")
    multiple_matrix = np.load("multiple_matrix.pkl")
    testPosition = np.load("testPosition.pkl")
    weights = np.load("weights.pkl")
    cf1 = np.load("cf1.pkl")
    cf2 = np.load("cf2.pkl")

    result, result_cf1, result_cf2 = ensemble_scoring(real_matrix, multiple_matrix, testPosition, weights, cf1, cf2)
    print(result, result_cf1, result_cf2)