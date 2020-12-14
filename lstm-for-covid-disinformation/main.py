# # LSTM demonstration
# An LSTM (Long Short-Term Memory) network is a special kind of recurrent neural network that attempts to address many
# of the problems with the basic model. See [here](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) for an
# excellent introduction and step-by-step analysis.
#
# ## Libraries
# We user [gensim](https://radimrehurek.com/gensim/index.html) to load the pre-trained word-embeddings; we use
# [torch](https://pytorch.org/), the PyTorch library, to implement the LSTM network; we use
# [pandas](https://pandas.pydata.org/pandas-docs/stable/index.html) to load the tsv file containing the dataset; we use
# [numpy](https://numpy.org/) to handle arrays; we use [tqdm](https://tqdm.github.io/) to display a graphic gauge of the
# progress as we train our network over the epochs; finally, we use
# [sklearn](https://scikit-learn.org/stable/index.html), SciKit-Learn, to compute the performance metrics.

import torch
import gensim as gensim
from tqdm import tqdm
from src.model import EnsembleModel
from src.util import TweetDataset, test, evaluateModel

# ## Pre-run initialisation
# We instantiate an object containing the pre-trained word embeddings and an object representing our classification
# model.

embeddings = "model.bin"
embeddings = gensim.models.KeyedVectors.load_word2vec_format(embeddings, binary=True)
embeddings_tensor = torch.FloatTensor(embeddings.vectors)
model = EnsembleModel(embeddings_tensor=embeddings_tensor)

# We split our dataset into training, validation and test fractions

training_set = TweetDataset(ds_loc="data/covid_training.tsv", embeddings_model=embeddings)
tr = len(training_set)
vd = int(tr * 0.10)
tr = tr - vd
training_set, validation_set = torch.utils.data.random_split(training_set, [tr, vd])
test_set = TweetDataset(ds_loc="data/covid_test_public.tsv", embeddings_model=embeddings)

# We set the configuration of the hyperparameters involved in the training run. For the training run, we will use
# [Adam](https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/) as our optimiser, which is a
# variant of Stochastic Gradient Descent, and Cross-Entropy as our cost function.

n_epochs = 15
learning_rate = 1e-4
optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

# `DataLoader` is a class that takes a `Dataset` object and returns a generator able to produce data batches. In this
# case, we are using the default batch size of 1

train_dl = torch.utils.data.DataLoader(training_set)
valid_dl = torch.utils.data.DataLoader(validation_set)
test_dl = torch.utils.data.DataLoader(test_set)

# ## Training run
# We train for a number of epochs equal to `n_epochs`, calculating the loss at each step, computing the gradient
# associated to each operation in the `EnsembleModel` object, and updating all model parameters. At each step, we store
# the values of the prediction and the target in order to compute the performance metrics at the end of each epoch. We
# perform a validation run at the end of each epoch in order to get an idea of how our model behaves before data it has
# never encountered and tune our hyperparameters accordingly.

for epoch in tqdm(range(1, n_epochs + 1), desc='Epochs', leave=True):
    tr_pred = []
    tr_target = []
    vd_pred = []
    vd_target = []
    for datum in tqdm(train_dl, desc='Training set', leave=False):
        input = datum[0]
        target = datum[1].squeeze(0)
        model.train()
        optimiser.zero_grad()
        prediction = model(input)
        loss = criterion(prediction, target)
        loss.backward()
        optimiser.step()

        tr_pred.append(prediction.squeeze())
        tr_target.append(target)
    prec, rec, f1 = test(tr_pred, tr_target)
    # print("Epoch:", epoch)
    # print("TRAINING: Precision: {0:.3g}\tRecall: {1:.3g}\tF1: {2:.3g}".format(prec, rec, f1))
    with torch.no_grad():
        for datum in tqdm(valid_dl, desc='Validation set', leave=False):
            input = datum[0]
            target = datum[1].squeeze(0)
            model.eval()
            optimiser.zero_grad()
            prediction = model(input)

            vd_pred.append(prediction.squeeze())
            vd_target.append(target)
        prec, rec, f1 = test(vd_pred, vd_target)
    # print("VALIDATION: Precision: {0:.3g}\tRecall: {1:.3g}\tF1: {2:.3g}".format(prec, rec, f1))

# ## Model evaluation
# Using the test fraction of our dataset, we evaluate our fully-trained, fully-tuned LSTM model. We use list
# comprehension to make things faster, selecting only the first element of each datapoint in the test set dataloader
# to make the prediction; recall each datapoint consists of a tuple containing the tweet and the annotation.

model.eval()
with torch.no_grad():
    test_prediction = [model(d[0]).squeeze() for d in tqdm(test_dl, desc="Test set")]
    test_target = [d[1] for d in test_dl]
    prec, rec, f1 = test(test_prediction, test_target, dataset=test_set, out=True)
    evaluateModel(model, test_prediction, test_target)
    print("TEST: Precision: {0:.4}\tRecall: {1:.4}\tF1: {2:.4}".format(prec, rec, f1))
