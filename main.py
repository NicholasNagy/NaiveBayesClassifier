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
        for k, v in from_merge.items():
            if k in to_merge:
                to_merge[k] += v
            else:
                to_merge[k] = v
        return to_merge

    @classmethod
    def filter_dict(cls, count_dict):
        l_to_del = [word for word, the_count in count_dict.items() if the_count == 1]
        for word in l_to_del:
            del count_dict[word]
        return count_dict

    def __smooth(self):
        for word, frequency in self.vocab.items():
            for class_label, class_dict in self.labels.items():
                if word in class_dict:
                    class_dict[word] += self.smoothing
                else:
                    class_dict[word] = self.smoothing

    def fit_params(self, smoothing=None, log_base=None):
        if self.fitted:
            print("Overriding previous fit")
        if smoothing is not None:
            self.smoothing = smoothing
        if log_base is not None:
            self.log_base = log_base

        if smoothing == 0.0 or log_base == 0.0:
            raise Exception(f'Smoothing ({smoothing}) or Log base ({log_base}) cannot be 0')
        self.__smooth()

        # for class_label, class_dict in self.labels.items():
        #     if len(class_dict) == len(self.vocab):
        #         print("ok")
        #     else:
        #         print("not ok")

        return self

    def train(self, the_df, data_label, class_label, do_filter=False):
        labels = dict()
        num_per_label = dict()
        for index, row in the_df.iterrows():
            # counts occurences of every word (obtained by splitting on ' ')
            word_dict = Counter(row[data_label].lower().split(' '))

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
        # for k, v in labels.items():
        #     print(f'{k}: {sum(v.values())}')
        # print(f'Num words in Vocab: {len(the_vocab)} Count of Words: {sum(the_vocab.values())}')
        self.labels, self.vocab = labels, the_vocab
        self.num_words_in_labels = {class_label: sum(class_dict.values()) for class_label, class_dict in labels.items()}

        tot_data_points = functools.reduce(lambda x, y: x+y, num_per_label.values())
        self.class_proportion = {label: num_data_points/tot_data_points
                                 for label, num_data_points in num_per_label.items()}
        return self

    def test_value(self, word_list, expected_value):
        prob_given_class = {class_label: [] for class_label, class_dict in self.labels.items()}
        for word in word_list:
            if word in self.vocab:
                for class_label, class_dict in self.labels.items():
                    prob_given_class[class_label].append(class_dict[word]/self.num_words_in_labels[class_label])

        results = {class_label: (math.log(self.class_proportion[class_label], self.log_base) +
                                 sum([math.log(value, self.log_base) for value in prob_values]))
                   for class_label, prob_values in prob_given_class.items()}
        evaluated_output = max(results.items(), key=operator.itemgetter(1))
        value = '%.2E' % Decimal(evaluated_output[1])
        predicted_class = evaluated_output[0]
        return {'predicted_output': predicted_class, 'predicted_value': value,
                'is_correct': 'correct' if predicted_class == expected_value else 'wrong'}

    def test(self, test_df, data_label, class_label, id_label):
        if self.smoothing == 0.0 or self.log_base == 0.0:
            raise Exception(f'Smoothing ({self.smoothing}) or Log base ({self.log_base}) cannot be 0')

        results = [{'output': self.test_value(row[data_label].lower().split(' '), row[class_label]),
                    'expected_output': row[class_label],
                    'id': row[id_label]}
                   for _, row in test_df.iterrows()]
        return results


if __name__ == '__main__':
    data = pd.read_table('covid_training.tsv')
    NB = NaiveBayes(0.01, 10).train(data, 'text', 'q1_label', True).fit_params()
    test_data = pd.read_table('covid_test_public.tsv')
    result = NB.test(test_data, 1, 2, 0)
    with open('result.txt', 'a') as f:
        for res in result:
            f.write(f'{res["id"]}  {res["output"]["predicted_output"]}  {res["output"]["predicted_value"]}  '
                    f'{res["expected_output"]}  {res["output"]["is_correct"]}\n')

