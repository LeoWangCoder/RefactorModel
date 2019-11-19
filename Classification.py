import numpy as np
from string import punctuation
from collections import Counter
from RefactoryRNN import RefactoryRNN
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def main():
    astPath ='/Users/mac/Downloads/refactortraindataset/MOVE_METHOD.txt'
    labelPath = '/Users/mac/Downloads/refactortraindataset/MOVE_METHODLabel.txt'
    with open (astPath, 'r') as f:
        abstractCode = f.read()
    with open(labelPath, 'r') as f:
        labels = f.read()
    # split the abstractCode and get all the text and words
    abstractCode = abstractCode.lower().replace(',', " ")
    all_text = ''.join([c for c in abstractCode if c not in punctuation])
    abstractCode_split = all_text.split('\n')
    all_text = ' '.join(abstractCode_split)
    words = all_text.split(' ')

    # Build a dictionary that maps words to integers

    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {word: i for i, word in enumerate(vocab, 1)}

    # store the tokenized abstractCode in abstractCode_inits
    abstractCode_ints = []
    for review in abstractCode_split:
        abstractCode_ints.append([vocab_to_int[word] for word in review.split()])

    print('Unique words: ', len((vocab_to_int)))
    print()

    # print tokens in first review
    print('Tokenized review: \n', abstractCode_ints[:1])

    # 1= refactoring, 0=no refactoring label conversion
    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == '1' else 0 for label in labels_split])

    review_lens = Counter([len(x) for x in abstractCode_ints])
    print("Zero-length abstractCode: {}".format(review_lens[0]))
    print("Maximum review length: {}".format(max(review_lens)))

    print('Number of abstractCode before removing error data: ', len(abstractCode_ints))

    ## remove any abstractCode/labels with zero length from the abstractCode_ints list.

    # get indices of any abstractCode with length 0
    non_zero_idx = [i for i, review in enumerate(abstractCode_ints) if len(review) != 0]

    # remove 0-length abstractCode and their labels
    abstractCode_ints = [abstractCode_ints[i] for i in non_zero_idx]
    encoded_labels = np.array([encoded_labels[i] for i in non_zero_idx])

    print('Number of abstractCode after removing error data: ', len(abstractCode_ints))

    #
    seq_length = 100
    features = pad_features(abstractCode_ints, seq_length=seq_length)

    ## test statements - do not change - ##
    assert len(features) == len(abstractCode_ints), "Your features should have as many rows as abstractCode."
    assert len(features[0]) == seq_length, "Each feature row should contain seq_length values."

    # print first 10 values of the first 30 batches
    print(features[0])


    #split train and test data
    split_frac = 0.8

    # split data into training, validation, and test data (features and labels, x and y)
    split_frac = 0.8
    split_idx = int(len(features) * split_frac)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x) * 0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

    ## print out the shapes of your resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # dataloaders
    batch_size = 100

    # SHUFFLE training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    # obtain one batch of training data
    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    print('Sample input size: ', sample_x.size())  # batch_size, seq_length
    print('Sample input: \n', sample_x)
    print()
    print('Sample label size: ', sample_y.size())  # batch_size
    print('Sample label: \n', sample_y)

    train_on_gpu = torch.cuda.is_available()

    if (train_on_gpu):
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    # Instantiate the model w/ hyperparams
    vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 400
    hidden_dim = 256
    n_layers = 2
    bidirectional = False

    net = RefactoryRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, bidirectional)
    print(net)
    # loss and optimization functions
    lr = 0.001

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # training params

    epochs = 2

    print_every = 10
    clip = 5  # gradient clipping

    # move model to GPU, if available
    if (train_on_gpu):
        net.cuda()

    net.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)
        counter = 0
        # batch loop
        for inputs, labels in train_loader:
            counter += 1
            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # backprop through the entire training history
            h = tuple([each.data for each in h])
            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

    print('train data is finished.')

    # Get test data loss and accuracy

    test_losses = []  # track loss
    num_correct = 0

    # init hidden state
    h = net.init_hidden(batch_size)

    net.eval()
    # iterate over test data
    for inputs, labels in test_loader:

        # Creating new variables for the hidden state, otherwise
        # backprop through the entire training history
        h = tuple([each.data for each in h])

        if (train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # get predicted outputs
        output, h = net(inputs, h)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())  # rounds to the nearest integer

        # compare predictions to true label
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    # -- stats! -- ##
    # avg test loss
    print("Test loss: {:.3f}".format(np.mean(test_losses)))

    # accuracy over all test data
    test_acc = num_correct / len(test_loader.dataset)
    print("Test accuracy: {:.3f}".format(test_acc))
    plt.figure()
    plt.plot(test_losses)
    plt.savefig("move_method.png", dpi=300, bbox_inches='tight')
    plt.show()

    # predict data to test the accuracy
    testInput = '/Users/mac/Downloads/testdata/EXTRACT_AND_MOVE_METHOD_Predict.txt'
    lines = []
    with open(testInput, 'r') as f:
        while True:
            line = f.readline()
            lines.append(line)
            if not line:
                break
    count = 0
    predictData = [ i for i in lines if i]
    for line in predictData:
        result = predict(net, line,vocab_to_int , seq_length)
        if(result == 1):
            count = count + 1

    print('%.4f' % (count/ len(lines)))


def pad_features(reviews_ints, seq_length):

    # getting the correct rows x cols shape
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    for i, row in enumerate(reviews_ints):
        features[i, -len(row):] = np.array(row)[:seq_length]

    return features


def tokenize_abstractCode(test_abstractCode, vocab_to_int):
    test_abstractCode = test_abstractCode.lower().replace(',', " ")  # lowercase
    # get rid of punctuation
    test_text = ''.join([c for c in test_abstractCode if c not in punctuation])

    # splitting by spaces
    test_words = test_text.split()

    # tokens
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])

    return test_ints


def predict(net, test_abstractCode, vocab_to_int, sequence_length=100):
    net.eval()

    # tokenize review
    test_ints = tokenize_abstractCode(test_abstractCode, vocab_to_int)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    train_on_gpu = torch.cuda.is_available()
    if (train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    #print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if (pred.item() == 1):
        return 1
    else:
        return 0


if __name__ == '__main__':
    main()
