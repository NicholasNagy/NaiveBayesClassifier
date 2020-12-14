import pandas as pd
from collections import Counter
import functools
import math
import operator
from decimal import Decimal


class NaiveBayes:

    def __init__(self, smoothing=0.0, log_base=0.0):
        self.labels = dict()
        self.num_words_in_labels = dict()
        self.class_proportion = dict()
        self.vocab = dict()
        self.smoothing = smoothing
        self.log_base = log_base
        self.fitted = False

    @classmethod
    def merge_dict(cls, from_merge, to_merge):
        # merges one dictionaries frequencies per word to the other and returns it
        for k, v in from_merge.items():
            if k in to_merge:
                to_merge[k] += v
            else:
                to_merge[k] = v
        return to_merge

    @classmethod
    def filter_dict(cls, count_dict):
        # removes the words for each labels' word to dictionary that have a count of only 1
        l_to_del = [word for word, the_count in count_dict.items() if the_count == 1]
        for word in l_to_del:
            del count_dict[word]
        return count_dict

    def __smooth(self):
        # smoothes out the frequencies for all the items in the vocabularies by adding 0.01
        for word, frequency in self.vocab.items():
            for class_label, class_dict in self.labels.items():
                if word in class_dict:
                    class_dict[word] += self.smoothing
                else:
                    class_dict[word] = self.smoothing

        # Adjust the # of words in each class (necessary after smoothing)
        self.num_words_in_labels = {label: sum([freq for word, freq in class_dict.items()])
                                    for label, class_dict in self.labels.items()}

    def fit_params(self, smoothing=None, log_base=None):
        # checks to make sure that the parameters are appropriately set to allow it to properly tested
        if self.fitted:
            print("Overriding previous fit")
        if smoothing is not None:
            self.smoothing = smoothing
        if log_base is not None:
            self.log_base = log_base

        if smoothing == 0.0 or log_base == 0.0:
            raise Exception(f'Smoothing ({smoothing}) or Log base ({log_base}) cannot be 0')
        self.__smooth()

        self.fitted = True

        return self

    def train(self, the_df, data_label, class_label, do_filter=False):
        # training that takes all the data and creates a dictionary where each key is a word, and each value is the
        # frequency of that word. This is held in the labels dict, which is a dictionary whose key is the label for
        # the dictionary it holds
        labels = dict()
        num_per_label = dict()
        for index, row in the_df.iterrows():
            # counts occurences of every word (obtained by splitting on ' ')
            word_dict = Counter(filter(None, row[data_label].lower().split(' ')))

            data_point_label = row[class_label]
            if not (data_point_label in labels):
                labels[data_point_label] = dict()
                num_per_label[data_point_label] = 0
            self.merge_dict(word_dict, labels[data_point_label])
            num_per_label[data_point_label] += 1

        if do_filter:
            labels = {the_class: self.filter_dict(class_dict) for the_class, class_dict in labels.items()}
        # perhaps the best way might not be to had the words to the dict, but this is faster than writing more
        # functionality
        the_vocab = functools.reduce(lambda l1, l2: self.merge_dict(l1[1], self.merge_dict(l2[1], dict())),
                                     labels.items())

        self.labels, self.vocab = labels, the_vocab
        self.num_words_in_labels = {class_label: sum(class_dict.values()) for class_label, class_dict in labels.items()}

        tot_data_points = functools.reduce(lambda x, y: x + y, num_per_label.values())
        self.class_proportion = {label: num_data_points / tot_data_points
                                 for label, num_data_points in num_per_label.items()}
        return self

    def test_value(self, word_list, expected_value):
        prob_given_class = {class_label: [] for class_label, class_dict in self.labels.items()}
        for word in word_list:
            if word in self.vocab:
                for class_label, class_dict in self.labels.items():
                    prob_given_class[class_label].append(class_dict[word] / self.num_words_in_labels[class_label])

        results = {class_label: (math.log(self.class_proportion[class_label], self.log_base) +
                                 sum([math.log(value, self.log_base) for value in prob_values]))
                   for class_label, prob_values in prob_given_class.items()}
        evaluated_output = max(results.items(), key=operator.itemgetter(1))
        value = '%.2E' % Decimal(evaluated_output[1])
        predicted_class = evaluated_output[0]
        return {'predicted_output': predicted_class, 'predicted_value': value,
                'is_correct': 'correct' if predicted_class == expected_value else 'wrong'}

    def test(self, test_df, data_label, class_label, id_label):
        # tests the model on each value, fails if required previous steps weren't completed
        if self.smoothing == 0.0 or self.log_base == 0.0:
            raise Exception(f'Smoothing ({self.smoothing}) or Log base ({self.log_base}) cannot be 0')
        if not self.fitted:
            raise Exception("Classifier needs to be fitted before testing")

        results = [{'output': self.test_value(row[data_label].lower().split(' '), row[class_label]),
                    'expected_output': row[class_label],
                    'id': row[id_label]}
                   for _, row in test_df.iterrows()]
        return results


class Stats:

    def __init__(self, results):
        self.results = results
        self.acc = self.accuracy()
        self.rec = self.recall()
        self.prec = self.precision()
        self.f1 = {label: (2*prec*self.rec[label])/(prec + self.rec[label])
                   for label, prec in self.prec.items()}

    def accuracy(self):
        return sum([1 if val["output"]["is_correct"] == 'correct' else 0 for val in self.results]) / len(self.results)

    def recall(self):
        labels = dict()
        for out in self.results:
            if not out["expected_output"] in labels:
                labels[out["expected_output"]] = {"tp": 0, "tp+fn": 0}
            labels[out["expected_output"]]["tp+fn"] += 1
            if out["output"]["is_correct"] == 'correct':
                labels[out["expected_output"]]["tp"] += 1
        return {label: the_result["tp"] / the_result["tp+fn"] for label, the_result in labels.items()}

    def precision(self):
        labels = dict()
        for out in self.results:
            if not out["output"]["predicted_output"] in labels:
                labels[out["output"]["predicted_output"]] = {"tp": 0, "tp+fp": 0}
            if out["output"]["is_correct"] == 'correct':
                labels[out["output"]["predicted_output"]]["tp"] += 1
            labels[out["output"]["predicted_output"]]["tp+fp"] += 1
        return {label: the_result["tp"] / the_result["tp+fp"] for label, the_result in labels.items()}

    def write_results(self, file_to_write):
        with open(file_to_write, 'w') as the_file:
            the_file.write(f'{self.acc}\n')
            the_file.write(f'{self.prec["yes"]}  {self.prec["no"]}\n')
            the_file.write(f'{self.rec["yes"]}  {self.rec["no"]}\n')
            the_file.write(f'{self.f1["yes"]}  {self.f1["no"]}\n')


def write_results(results, file_to_write):
    with open(file_to_write, 'w') as f:
        for res in results:
            f.write(f'{res["id"]}  {res["output"]["predicted_output"]}  {res["output"]["predicted_value"]}  '
                    f'{res["expected_output"]}  {res["output"]["is_correct"]}\n')


if __name__ == '__main__':
    data = pd.read_table('covid_training.tsv')
    test_data = pd.read_table('covid_test_public.tsv')
    NB = NaiveBayes(0.01, 10).train(data, 'text', 'q1_label', False).fit_params()
    result = NB.test(test_data, 1, 2, 0)
    write_results(result, "trace_NB-BOW-OV.txt")
    stat = Stats(result)
    stat.write_results("eval_NB-BOW-OV.txt")

    NB_filtered = NaiveBayes(0.01, 10).train(data, 'text', 'q1_label', True).fit_params()
    result_filter = NB_filtered.test(test_data, 1, 2, 0)
    write_results(result_filter, "trace_NB-BOW-FV.txt")
    stat = Stats(result_filter)
    stat.write_results("eval_NB-BOW-FV.txt")
