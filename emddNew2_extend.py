# -*-coding:utf-8-*-
from __future__ import division
import scipy.io
import scipy.optimize
import numpy
import math
import random
import pprint
import pandas as pd
import sys
from collections.abc import Sequence
from collections import namedtuple
from enum import Enum
from functools import reduce
import pickle

EMDDResult = namedtuple('EMDDResult', ['target', 'scale', 'density'])
PredictionResult = namedtuple('PredictionResult', [
    'bags',
    'actual_positive_bags',
    'actual_negative_bags',
    'true_positives',
    'true_negatives',
    'false_positives',
    'false_negatives',
    'accuracy',
    'precision',
    'recall'
])

pp = pprint.PrettyPrinter(indent=4)

prediction_threshold = 0.6
training_threshold = 0.1

cross_validation_k = 10


class Aggregate(Enum):
    min = 1
    max = 2
    avg = 3

class EMDD:
    def __init__(self, training_data):
        self.training_data = training_data
    @staticmethod
    def predict(results, bags, aggregate=Aggregate.avg):
        if aggregate == Aggregate.max:
            result = sorted(results, key=lambda r: r.density)[::-1][0]
        elif aggregate == Aggregate.min:
            result = sorted(results, key=lambda r: r.density)[0]
        else:
            result_sum = reduce(
                lambda x, y: EMDDResult(
                    target=x.target + y.target,
                    scale=x.scale + y.scale,
                    density=x.density + y.density
                ), results
            )
            result = EMDDResult(
                target=result_sum.target / len(results),
                scale=result_sum.scale / len(results),
                density=result_sum.density / len(results)
            )
        print("Using result", result, "for prediction")
        return EMDD.classify(bags, result.target, result.scale)

    @staticmethod
    def predict_n(results_n,bags):
        targets_scales = []
        for training_results in results_n:
            prediction_result = EMDD.predict_test(results=training_results, aggregate=Aggregate.avg)
            targets_scales.append([prediction_result.target,prediction_result.scale])
        return EMDD.classify_test(bags,targets_scales)

    @staticmethod
    def predict_test(results, aggregate=Aggregate.avg):
        if aggregate == Aggregate.max:
            result = sorted(results, key=lambda r: r.density)[::-1][0]
        elif aggregate == Aggregate.min:
            result = sorted(results, key=lambda r: r.density)[0]
        else:
            result_sum = reduce(
                lambda x, y: EMDDResult(
                    target=x.target + y.target,
                    scale=x.scale + y.scale,
                    density=x.density + y.density
                ), results
            )
            result = EMDDResult(
                target=result_sum.target / len(results),
                scale=result_sum.scale / len(results),
                density=result_sum.density / len(results)
            )
        return result


    @staticmethod
    def classify_test(bags, targets_scales):
        total_bags = len(bags)
        actual_positive_bags = len([bag for bag in bags if bag.is_positive()])
        actual_negative_bags = total_bags - actual_positive_bags
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        positive_bag_indexes = []
        probabilities = []
        probabilities_n = []
        for i in range(0, len(bags)):
            instances = []
            instance_probabilities = []
            temp_probabilities = numpy.zeros(len(bags[i].instances))
            n = 0
            for target_scale in targets_scales:
                target = target_scale[0]
                scale = target_scale[1]
                if n > 0:
                    temp_probabilities = EMDD.positive_instance_probability(target, scale, bags[i].instances)
                    probabilities_n = list(map(lambda x: x[0] + x[1], zip(probabilities_n, temp_probabilities)))
                    n += 1
                else:
                    n += 1
                    probabilities_n = temp_probabilities
                    temp_probabilities = EMDD.positive_instance_probability(target, scale, bags[i].instances)
                    probabilities_n = list(map(lambda x: x[0] + x[1], zip(probabilities_n, temp_probabilities)))
            probabilities = list(map(lambda x: x / n, probabilities_n))
            for j in range(0, len(probabilities)):
                if probabilities[j] > prediction_threshold:
                    instances.append(j)
                    instance_probabilities.append(probabilities[j])
            if len(instances) > 0:
                if bags[i].is_positive():
                    true_positives += 1
                else:
                    false_positives += 1
                positive_bag_indexes.append(i)
            else:
                if bags[i].is_positive():
                    false_negatives += 1
                else:
                    true_negatives += 1
        if true_positives == 0 and false_positives == 0:
            precision = 1
        else:
            precision = true_positives / (true_positives + false_positives)
        if true_positives == 0 and false_negatives == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)
        return PredictionResult(
            bags=positive_bag_indexes,
            actual_positive_bags=actual_positive_bags,
            actual_negative_bags=actual_negative_bags,
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            accuracy=(true_positives + true_negatives) / total_bags,
            precision=precision,
            recall=recall
        )
    @staticmethod
    def classify(bags, target, scale):
        total_bags = len(bags)
        actual_positive_bags = len([bag for bag in bags if bag.is_positive()])
        actual_negative_bags = total_bags - actual_positive_bags
        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        positive_bag_indexes = []
        for i in range(0, len(bags)):
            instances = []
            instance_probabilities = []
            probabilities = EMDD.positive_instance_probability(target, scale, bags[i].instances)
            for j in range(0, len(probabilities)):
                if probabilities[j] > prediction_threshold:
                    instances.append(j)
                    instance_probabilities.append(probabilities[j])
            if len(instances) > 0:
                if bags[i].is_positive():
                    true_positives += 1
                else:
                    false_positives += 1
                positive_bag_indexes.append(i)
            else:
                if bags[i].is_positive():
                    false_negatives += 1
                else:
                    true_negatives += 1
        if true_positives == 0 and false_positives == 0:
            precision = 1
        else:
            precision = true_positives / (true_positives + false_positives)
        if true_positives == 0 and false_negatives == 0:
            recall = 0
        else:
            recall = true_positives / (true_positives + false_negatives)
        return PredictionResult(
            bags=positive_bag_indexes,
            actual_positive_bags=actual_positive_bags,
            actual_negative_bags=actual_negative_bags,
            true_positives=true_positives,
            true_negatives=true_negatives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            accuracy=(true_positives + true_negatives) / total_bags,
            precision=precision,
            recall=recall
        )

    @staticmethod
    def quick_sort(arr, low, high):
        i = low
        j = high
        if i >= j:
            return arr
        temp = arr[i]
        while i < j:
            while i < j and arr[j][0].accuracy >= temp[0].accuracy:
                j = j - 1
            arr[i] = arr[j]
            while i < j and arr[i][0].accuracy <= temp[0].accuracy:
                i = i + 1
            arr[j] = arr[i]
        arr[i] = temp
        EMDD.quick_sort(arr, low, i - 1)
        EMDD.quick_sort(arr, j + 1, high)
        return arr

    def train(self, m, perform_scaling=False):
        results = []
        accuracy_sum = 0
        precision_sum = 0
        recall_sum = 0
        best_result = []
        n_h = []
        for partition_number in range(0, cross_validation_k):
            for bag in self.training_data.training_bags:
                for instance in bag.instances:
                    instance.used_as_target = False
            partition_results = []
            validation_and_training_set = self.training_data.validation_and_training_set(partition_number)
            validation_set = validation_and_training_set["validation_set"]
            training_set = validation_and_training_set["training_set"]
            positive_training_bags = [bag for bag in training_set if bag.is_positive()]
            random.shuffle(positive_training_bags)    # 将序列打乱
            k = 10
            random_positive_bags = list(map(lambda x: positive_training_bags[x], range(0, k)))
            total_instances = sum([len(bag.instances) for bag in random_positive_bags])
            print("Cross-validation on partition", (partition_number + 1), "with", total_instances, "instances")
            instance_number = 1
            for random_positive_bag in random_positive_bags:
                for instance in random_positive_bag:
                    partition_results.append(EMDD.run(
                        perform_scaling,
                        training_set,
                        instance.features,
                        numpy.full(instance.features.size, 0.1)
                    ))
                    instance_number += 1
            accuracy = 0
            precision = 0
            recall = 0
            tem_ = []
            for partition_result in partition_results:
                prediction_result = EMDD.classify(validation_set, partition_result.target, partition_result.scale)
                tem_.append([prediction_result,partition_result])
                if prediction_result.accuracy > accuracy:
                    accuracy = prediction_result.accuracy
                    precision = prediction_result.precision
                    recall = prediction_result.recall
                    best_result = partition_result
            """
            n_h    
                ____Sherwin
            """
            sort_partition_results = EMDD.quick_sort(tem_, 0, len(tem_) - 1)
            accuracy_sum += accuracy
            precision_sum += precision
            recall_sum += recall
            print("Partition {}: Density: {} Accuracy: {} Precision: {} Recall: {}".format(
                partition_number, best_result.density, accuracy, precision, recall
            ))
            results.append(best_result)
            tem = []
            for t in sort_partition_results[-3:]:
                tem.append(t[1])
            n_h.append(tem)
        print("Across {}-fold cross-validation:".format(cross_validation_k))
        print("Average accuracy:", (accuracy_sum / cross_validation_k))
        print("Average precision:", (precision_sum / cross_validation_k))
        print("Average recall:", (recall_sum / cross_validation_k))
        return results,n_h

    @staticmethod
    def run(perform_scaling, bags, target, scale):
        density_difference = math.inf
        previous_density = math.inf
        best_density = math.inf
        density = 0

        while density_difference > training_threshold:
            optimal_instances = []
            for bag in bags:
                probabilities = EMDD.positive_instance_probability(target, scale, bag.instances)
                # print("run", run, "positive", bag.is_positive(), "bag", bag.index, "instance", numpy.argmax(probabilities), "probability", probabilities[numpy.argmax(probabilities)])
                optimal_instances.append(bag.instances[numpy.argmax(probabilities)])  #argmax返回的是最大数的索引
            if perform_scaling:
                params = numpy.concatenate((target, scale))
                lower_bound = numpy.zeros(2 * target.size)
                upper_bound = numpy.concatenate((numpy.ones(target.size), numpy.full(target.size, numpy.inf)))
            else:
                params = target
                lower_bound = numpy.zeros(target.size)
                upper_bound = numpy.ones(target.size)
            bounds = tuple(map(lambda x: x, zip(lower_bound, upper_bound)))
            result = scipy.optimize.minimize(
                fun=EMDD.diverse_density,
                jac=EMDD.diverse_density_gradient,
                x0=params,
                args=optimal_instances,
                bounds=bounds,
                method='L-BFGS-B',
                options={
                    'ftol': 1.0e-06,
                    'maxfun': 100000,
                    'maxiter': 2000,
                }
            )
            params = result.x
            if perform_scaling:
                target = params[0:target.size]
                scale = params[target.size:2 * target.size]
            else:
                target = params
                scale = numpy.ones(target.size)
            density = result.fun
            if density < best_density:
                density_difference = best_density - density
                best_density = density
                previous_density = density
            else:
                density_difference = previous_density - density
                if density_difference != 0:
                    density_difference = 2 * training_threshold
                    previous_density = density
        return EMDDResult(target=target, scale=scale, density=density)

    @staticmethod
    def positive_instance_probability(target, scale, instances):
        x = numpy.tile(target, (len(instances), 1))
        s = numpy.tile(scale, (len(instances), 1))
        b_ij = numpy.array(list(map(lambda i: i.features, instances)))
        # print(numpy.square(s) * numpy.square(b_ij - x))
        distances = numpy.mean(numpy.square(s) * numpy.square(b_ij - x), 1)
        return numpy.exp(numpy.negative(distances))

    @staticmethod
    def diverse_density(params, instances):
        num_features = instances[0].features.size
        scaling = params.size != num_features
        if not scaling:
            target = params
            scale = numpy.ones(num_features)
        else:
            target = params[0:num_features]
            scale = params[num_features:2 * num_features]
        p = EMDD.positive_instance_probability(target, scale, instances)
        density = 0
        for i in range(0, len(instances)):
            if instances[i].bag.is_positive():
                if p[i] == 0:
                    p[i] = 1.0e-10
                density -= math.log(p[i])
            else:
                if p[i] == 1:
                    p[i] = 1 - 1.0e-10
                density -= math.log(1 - p[i])
        return density

    @staticmethod
    def diverse_density_gradient(params, instances):
        num_features = instances[0].features.size
        scaling = params.size != num_features
        if not scaling:
            target = params
            scale = numpy.ones(num_features)
            gradient = numpy.zeros(num_features)
        else:
            target = params[0:num_features]
            scale = params[num_features:2 * num_features]
            gradient = numpy.zeros(2 * num_features)
        p = EMDD.positive_instance_probability(target, scale, instances)
        for d in range(0, num_features):
            for i in range(0, len(instances)):
                if instances[i].bag.is_positive():
                    if p[i] == 0:
                        p[i] = 1.0e-10
                    gradient[d] -= (2 / num_features) * \
                                   (scale[d] ** 2) * \
                                   (instances[i].features[d] - target[d])
                    if scaling:
                        gradient[d + num_features] += (2 / num_features) * scale[d] * \
                                                      ((instances[i].features[d] - target[d]) ** 2)
                else:
                    if p[i] == 1:
                        p[i] = 1 - 1.0e-10
                    gradient[d] += (1 / (1 - p[i])) * \
                                   (2 / num_features) * \
                                   (scale[d] ** 2) * \
                                   (instances[i].features[d] - target[d])
                    if scaling:
                        gradient[d + num_features] -= (1 / (1 - p[i])) * \
                                                      (2 / num_features) * scale[d] * \
                                                      ((instances[i].features[d] - target[d]) ** 2)
        return gradient

class MatlabTrainingData:

    def __init__(self, file_name, handler):
        if file_name.endswith('.mat'):
            data = handler(scipy.io.loadmat(file_name))
        else:
            data = handler(pd.read_csv(file_name, low_memory=False))

        self.training_bags = data["training_bags"]

        # setting up partitions for k-fold cross-validation
        training_bags = self.training_bags[:]
        random.shuffle(training_bags)
        bags_per_partition = math.floor(len(training_bags) / cross_validation_k)
        bag_index = 0
        self.partitions = {}
        for i in range(0, cross_validation_k):
            self.partitions[i] = []
            for j in range(bag_index, min(bag_index + bags_per_partition, len(training_bags))):
                self.partitions[i].append(training_bags[j])
                bag_index += 1
        self.test_bags = data["test_bags"]

    def validation_and_training_set(self, partition_number):
        return {
            "validation_set": self.partitions[partition_number],
            "training_set": [bag for partition in list(
                map(lambda x: self.partitions[x], [j for j in range(0, cross_validation_k) if j != partition_number])
            ) for bag in partition]
        }
class Bags(Sequence):

    def __init__(self, bags):
        self.bags = bags

    def __getitem__(self, index):
        return self.bags[index]

    def __len__(self):
        return len(self.bags)


class Bag(Sequence):

    def __init__(self, index, label):
        self.index = index
        self.instances = []
        self.label = label

    def add_instance(self, instance):

        self.instances.append(Instance(instance, self))

    def is_positive(self):
        return self.label == 1

    def __getitem__(self, index):
        return self.instances[index]

    def __len__(self):
        return len(self.instances)

class Instance:
    """Describes an instance in a bag"""

    def __init__(self, features, bag):
        self.bag = bag
        self.features = features
        self.used_as_target = False

def load_data(mat, bags_key, labels_key):
    bags = []

    mat_bags = mat[bags_key]
    it_bags = numpy.nditer(mat_bags, ["refs_ok", "c_index"])
    while not it_bags.finished:
        mat_bag = mat_bags[0][it_bags.index]
        label = mat[labels_key][0][it_bags.index]

        bag = Bag(it_bags.index, label)
        i = 0
        for instance in mat_bag:
            bag.add_instance(instance)
            inst_list = list(map(lambda x: str(x), instance.tolist()))
            #print("INST_{}_{},BAG_{},{},{}".format(i, it_bags.index, it_bags.index, label, ",".join(inst_list)))
            i += 1
        bags.append(bag)
        it_bags.iternext()
    return bags

def load_synth_data(mat):
    bags = load_data(mat, "bag", "labels")
    return {
        "training_bags": bags,
        "test_bags": bags
    }

def load_dr_data(mat):
    return {
        "training_bags": load_data(mat, "bag", "labels"),
        "test_bags": load_data(mat, "testBags", "testlabels"),
    }

def load_musk_data(mat):
    bag_map = {}
    it_bag_ids = numpy.nditer(mat["bag_ids"], ["refs_ok", "c_index"])

    while not it_bag_ids.finished:

        bag_index = mat["bag_ids"][0][it_bag_ids.index]

        if bag_index not in bag_map:
            bag_map[bag_index] = Bag(bag_index, -1)

        bag = bag_map[bag_index]
        bag.add_instance(numpy.array(mat["features"][it_bag_ids.index].A[0]))
        if not bag.is_positive() and mat["labels"].A[0][it_bag_ids.index] == 1:
            bag.label = 1
        it_bag_ids.iternext()

    return {
        "training_bags": list(bag_map.values()),
        "test_bags": list(bag_map.values())
    }

def load_csv_data(csv_reader):
    """
    process the csv file
            ____Sherwin
    """
    bag_map = {}
    it_bag_ids = numpy.nditer(csv_reader["id"], ["refs_ok", "c_index"])

    while not it_bag_ids.finished:

        bag_index = csv_reader["id"][it_bag_ids.index]

        if bag_index not in bag_map:
            bag_map[bag_index] = Bag(bag_index, -1)

        bag = bag_map[bag_index]
        features = [csv_reader['nhet'][it_bag_ids.index],csv_reader['cnlr.median'][it_bag_ids.index],csv_reader['mafR'][it_bag_ids.index],
                    csv_reader['mafR.clust'][it_bag_ids.index],csv_reader['start'][it_bag_ids.index],csv_reader['end'][it_bag_ids.index],
                    csv_reader['cf.em'][it_bag_ids.index],csv_reader['tcn.em'][it_bag_ids.index],csv_reader['lcn.em'][it_bag_ids.index]]
        bag.add_instance(numpy.array(features))
        if not bag.is_positive() and csv_reader["label"][it_bag_ids.index] == 1:
            bag.label = 1
        it_bag_ids.iternext()

    return {
        "training_bags": list(bag_map.values()),
        "test_bags": list(bag_map.values())
    }


def load_animal_data(mat):
    bag_map = {}
    it_bag_ids = numpy.nditer(mat["bag_ids"], ["refs_ok", "c_index"])
    while not it_bag_ids.finished:
        bag_index = mat["bag_ids"][0][it_bag_ids.index]
        if bag_index not in bag_map:
            bag_map[bag_index] = Bag(bag_index, -1)

        bag = bag_map[bag_index]
        instance = numpy.array(mat["features"][it_bag_ids.index].A[0])

        label = mat["labels"].A[0][it_bag_ids.index]
        if label != 1:
            label = 0
        bag.add_instance(instance)
        inst_list = list(map(lambda x: str(x), instance.tolist()))
        #print("INST_{}_{},BAG_{},{},{}".format(len(bag.instances), bag_index, bag_index, label, ",".join(inst_list)))
        if not bag.is_positive() and label == 1:
            bag.label = 1
        it_bag_ids.iternext()
    return {
        "training_bags": list(bag_map.values()),
        "test_bags": list(bag_map.values())
    }

def load_fake_data(mat):
    positive_instance = numpy.array([1, 0, 1, 1, 0])
    negative_instance = numpy.array([0, 1, 0, 0, 0])
    bags = []
    for i in range(0, 100):
        bag = Bag(i, i % 2)
        if i % 2 == 0:
            for j in range(0, 10):
                bag.add_instance(negative_instance)
        else:
            num_positive = random.randrange(0, 5) + 1
            for j in range(0, num_positive):
                bag.add_instance(positive_instance)

            for j in range(0, 10 - num_positive):
                bag.add_instance(negative_instance)
        bags.append(bag)
        print("bag", i, bag.is_positive(), "has instances")
        for instance in bag.instances:
            print("    instance", instance.features)
    return {
        "training_bags": bags,
        "test_bags": bags
    }

def get_training_data(data_set):
    if data_set == "fake":
        training_data = MatlabTrainingData('F:/shenzhen/Sherwin/em-dd-master/training-data/musk1norm_matlab.mat', load_fake_data)
    elif data_set == "csv":
        training_data = MatlabTrainingData('F:/shenzhen/Sherwin/HRD/all_scaler.csv', load_csv_data)
    elif data_set == "musk1":
        training_data = MatlabTrainingData('C:/Users/Sherwin/Desktop/em-dd-master/training-data/musk1norm_matlab.mat', load_musk_data)
    elif data_set == "musk2":
        training_data = MatlabTrainingData('C:/Users/Sherwin/Desktop/em-dd-master/training-data/musk2norm_matlab.mat', load_musk_data)
    elif data_set == "synth1":
        training_data = MatlabTrainingData('C:/Users/Sherwin/Desktop/em-dd-master/training-data/synth_data_1.mat', load_synth_data)
    elif data_set == "synth4":
        training_data = MatlabTrainingData('C:/Users/Sherwin/Desktop/em-dd-master/training-data/synth_data_4.mat', load_synth_data)
    elif data_set == "dr":
        training_data = MatlabTrainingData('C:/Users/Sherwin/Desktop/em-dd-master/training-data/DR_data.mat', load_dr_data)
    elif data_set == "elephant":
        training_data = MatlabTrainingData('C:/Users/Sherwin/Desktop/em-dd-master/training-data/elephant_100x100_matlab.mat', load_animal_data)
    elif data_set == "fox":
        training_data = MatlabTrainingData('C:/Users/Sherwin/Desktop/em-dd-master/training-data/fox_100x100_matlab.mat', load_animal_data)
    elif data_set == "tiger":
        training_data = MatlabTrainingData('C:/Users/Sherwin/Desktop/em-dd-master/training-data/tiger_100x100_matlab.mat', load_animal_data)
    else:
        raise ValueError("Unknown data set. Use one of fake, musk1, musk2, synth1, synth4, dr, elephant, tiger, or fox")

    return training_data

# if len(sys.argv) == 1:
#     raise ValueError("Please provide the name of the data set as an argument")
# training_data = get_training_data(sys.argv[1])
training_data = get_training_data('csv')

emdd = EMDD(training_data)
n = -3
training_results,training_results_n = emdd.train(n,perform_scaling=True)
pickle.dump([training_results,training_results_n], open('trained_EMDDN2.pkl', 'wb'))#保存模型
test_bags = training_data.training_bags
total_bags = len(test_bags)
print(training_results,training_results_n)
prediction_result = EMDD.predict(results=training_results, bags=test_bags, aggregate=Aggregate.avg)
prediction_result_n = EMDD.predict_n(results_n=training_results_n,bags=test_bags)

print("Classifying against test set")
print("")
print("Threshold", prediction_threshold)
print("Total bags", total_bags)
print("Total positive bags", prediction_result.actual_positive_bags,prediction_result_n.actual_positive_bags)
print("Total negative bags", prediction_result.actual_negative_bags,prediction_result_n.actual_negative_bags)
print("Total predicted positive bags", len(prediction_result.bags),len(prediction_result_n.bags))
print("Total predicted negative bags", total_bags - len(prediction_result.bags),total_bags - len(prediction_result_n.bags))
print("True positives", prediction_result.true_positives,prediction_result_n.true_positives)
print("False positives", prediction_result.false_positives,prediction_result_n.false_positives)
print("True negatives", prediction_result.true_negatives,prediction_result_n.true_negatives)
print("False negatives", prediction_result.false_negatives,prediction_result_n.false_negatives)
print("Accuracy", prediction_result.accuracy,prediction_result_n.accuracy)
print("Precision", prediction_result.precision,prediction_result_n.precision)
print("Recall", prediction_result.recall,prediction_result_n.recall)
