# ## Helper classes and functions
# A `Lexicon` object contains the vocabulary of an embeddings model along with dictionaries mapping each term to a index
# and each index to a term. `tensorFomSentence` takes string representing a tweet and returns a representation as a
# sequence of indices taken from the embeddings model. `TweetDataset` is an inherited class used to fetch datapoints
# from the dataset and return them as model-ready tuples of tweets and annotations.`test` is a function that
# returns the metrics of precision, recall and F1 given a list of predictions and targets. Finally, `evaluateModel` is a
# function that produces the evaluation file of the model

import torch
import numpy as np
from pandas import read_csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class Lexicon:
    """Helper class. Collects vocabulary of embeddings model into key-value and value-key pairs"""

    def __init__(self, embeddings):
        self.word2index = {token: token_index for token_index, token in enumerate(embeddings.index2word)}
        self.index2word = embeddings.index2word
        self.n_words = len(embeddings.index2word)


def tensorFromSentence(lexicon, sentence):
    """Returns a PyTorch tensor object from a string

    In: embeddings model and a sentence string
    Out: long tensor object

    """
    index_list = []
    sentence = sentence.split(' ')
    for word in sentence:
        try:
            index = lexicon.word2index[word.lower()]
        except KeyError:
            index = 0
        index_list.append(index)
    return torch.tensor(index_list, dtype=torch.long)


class TweetDataset(torch.utils.data.Dataset):
    """
    Inherited class used to fetch datapoints from the dataset and return them as model-ready tuples of tweets and
    annotations
    """
    def __init__(self, ds_loc, embeddings_model):
        self.df = read_csv(ds_loc, sep="\t", header=None)
        self.lexicon = Lexicon(embeddings_model)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        tweet = self.df.iloc[index, 1]
        tweet = tensorFromSentence(self.lexicon, tweet)
        tweet_id = self.df.iloc[index, 0]
        annotation = self.df.iloc[index, 2]
        annotation = [1] if annotation == 'yes' else [0]
        annotation = torch.tensor(annotation, dtype=torch.long)
        return tweet, annotation, tweet_id


def pred(p):
    """
    Helper function
    :param p: 2D tensor
    :return: int
    """
    return 1 if p[0] < p[1] else 0


def test(prediction, target, dataset=None, out=False):
    """
    Returns evaluation metrics
    :param target: tensor containing target values
    :param prediction: tensor containing prediction values
    :param dataset: (optional, required to produce trace file) test dataset
    :param out: (optional, required to produce trace file) whether to produce trace or not
    :return: tuple containing precision, recall, and f1 values
    """
    t = np.array([x.item() for x in target])
    p = np.array([pred(x) for x in prediction])
    if out is not False:
        if dataset is None:
            raise ValueError("Dataset is needed to retrieve tweet ids!")
        with open("trace_LSTM.txt", "w") as file:
            for i in range(len(dataset)):
                tweet_id = dataset[i][2]
                prediction_text = "yes" if pred(prediction[i]) == 1 else "no"
                prediction_proba = prediction[i][1].item() if prediction_text == "yes" else prediction[i][0].item()
                target_text = "yes" if target[i] == 1 else "no"
                outcome = "correct" if prediction_text == target_text else "wrong"
                line = """{}  {}  {:.4}  {}  {}\n""".format(tweet_id, prediction_text, prediction_proba, target_text, outcome)
                file.write(line)
        print("Trace file produced: 'trace_LSTM.txt'")
    try:
        pre = precision_score(t, p)
        rec = recall_score(t, p)
        f1 = f1_score(t, p)
    except ValueError:
        f1 = pre = rec = 0
    return pre, rec, f1


def evaluateModel(model, prediction, target):
    """
    Produces output file containing model evaluation
    :param model: lstm torch model
    :param prediction: tensor containing prediction values
    :param target: tensor containing target values
    :return: None
    """
    t_yes = np.array([x.item() for x in target])
    t_no = np.array([(lambda x: 1 if x == 0 else 0)(n) for n in t_yes])
    p_yes = np.array([pred(x) for x in prediction])
    p_no = np.array([(lambda x: 1 if x == 0 else 0)(n) for n in p_yes])
    try:
        pre_yes = precision_score(t_yes, p_yes)
        rec_yes = recall_score(t_yes, p_yes)
        f1_yes = f1_score(t_yes, p_yes)
        acc = accuracy_score(t_yes, p_yes)

        pre_no = precision_score(t_no, p_no)
        rec_no = recall_score(t_no, p_no)
        f1_no = f1_score(t_no, p_no)
    except ValueError:
        f1_yes = pre_yes = rec_yes = 0
        f1_no = pre_no = rec_no = 0
    with open("eval_lstm.txt", "w") as file:
        file.write("{:.4}\n".format(acc))
        file.write("{:.4}  {:.4}\n".format(pre_yes, pre_no))
        file.write("{:.4}  {:.4}\n".format(rec_yes, rec_no))
        file.write("{:.4}  {:.4}\n".format(f1_yes, f1_no))
    print("Evaluation file produced: 'eval_lstm.txt'")
