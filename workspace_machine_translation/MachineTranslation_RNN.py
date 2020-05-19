import random
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import nltk

from torch.utils.tensorboard import SummaryWriter

import Utils as utils


class EncoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, x_vocabulary_size, max_seq_length):
        super(EncoderRNN, self).__init__()

        self.max_seq_length = max_seq_length

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.x_vocabulary_size = x_vocabulary_size
        self.num_layer = 1  # one layer
        # torch embedding is used to store embeddings for words, basically an embedding it's a fixed size vector
        # representation of a word (kind of hashtable? loosely similar)
        #   word index (word2idx output) -> [ embedding ] -> word vector
        self.embedding = nn.Embedding(self.x_vocabulary_size, self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size, num_layers=self.num_layer,
                        hidden_size=self.hidden_size,
                        # nonlinearity functions are fixed for LSTM
                        batch_first=False  # batch is in the first position
                        )
        # the Encoder fully connected output HAS no use! you can see it also from training
        #self.out_fc = nn.Linear(in_features=self.hidden_size, out_features=self.x_vocabulary_size) # num words
        #self.out_fc.weight.data.normal_(0, 0.01)
        #self.out_fc.bias.data.zero_()

        #self.lstm.weight_ih_l0.data.normal_(0, 0.01)
        #self.lstm.weight_hh_l0.data.normal_(0, 0.01)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        #torch.nn.init.xavier_normal_(self.embedding) # <-- NON ha senso dato che non è trainabile!
        #torch.nn.init.xavier_normal_(self.out_fc.weight)
        torch.nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        torch.nn.init.xavier_normal_(self.lstm.weight_hh_l0)

    # input: x shape [batch_size, seq_len, input vector size], batch_size must be 1
    #        input vector size must be 1, since it's only a number/index
    # encoder do not have any loss since the input is always given and output is always ignored
    # HOWEVER, it will be trained with backpropagation trough time with the loss computed by the decoder
    def forward(self, x):
        #assert x.shape[0] == 1, "batch size must be 1"
        #assert x.shape[2] == 1, "input vector size must be 1"
        #assert x.shape[1] <= max_seq_length, "max length must be respected " + str(x.shape[0]) + " not lower than " + str(max_seq_length)

        batch_size, seq_len, input_vector_size = x.shape
        x = x.reshape(seq_len, batch_size, input_vector_size)

        # this initialization is random, but probably it's a good idea to normalize also the input x,
        # if something doesn't work
        h = torch.zeros((self.num_layer, batch_size, self.hidden_size)) # torch.empty((self.num_layer, batch_size, self.hidden_size)).normal_(mean=0.0, std=0.01)
        c = torch.zeros((self.num_layer, batch_size, self.hidden_size)) # torch.empty((self.num_layer, batch_size, self.hidden_size)).normal_(mean=0.0, std=0.01)
        h = h.cuda()
        c = c.cuda()

        # embedding input(*) - LongTensor of arbitrary shape containing the indices to extract
        # Output: (∗,H), where * is the input shape and H=embedding_dim
        for time_step in range(seq_len):
            input = self.embedding(x[time_step, 0, 0]).view(1, 1, -1) # this view is necessary to transform shape [256] -> [1,1,256]
            input = F.relu(input)
            out, (h, c) = self.lstm(input, (h, c))

        return out, (h, c)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, embedding_size, y_vocabulary_size, max_seq_length, loss_criterion, target_SOS_idx, target_EOS_idx):
        super(DecoderRNN, self).__init__()

        self.max_seq_length = max_seq_length
        self.loss_criterion = loss_criterion
        self.target_SOS_idx = target_SOS_idx
        self.target_EOS_idx = target_EOS_idx

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.y_vocabulary_size = y_vocabulary_size
        self.num_layer = 1  # one layer
        # torch embedding is used to store embeddings for words, basically an embedding it's a fixed size vector
        # representation of a word (kind of hashtable? loosely similar)
        #   word index (word2idx output) -> [ embedding ] -> word vector
        self.embedding = nn.Embedding(self.y_vocabulary_size, self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size, num_layers=self.num_layer,
                        hidden_size=self.hidden_size,
                        # nonlinearity functions are fixed for LSTM
                        batch_first=False  # batch is in the first position
                        )
        self.out_fc = nn.Linear(in_features=self.hidden_size, out_features=self.y_vocabulary_size) # num words

        #self.out_fc.weight.data.normal_(0, 0.01)
        self.out_fc.bias.data.zero_()

        #self.lstm.weight_ih_l0.data.normal_(0, 0.01)
        #self.lstm.weight_hh_l0.data.normal_(0, 0.01)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        torch.nn.init.xavier_normal_(self.out_fc.weight)
        torch.nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        torch.nn.init.xavier_normal_(self.lstm.weight_hh_l0)

        self.log_softmax = nn.LogSoftmax(dim=1)

    # input: x shape [batch_size, seq_len, input vector size], batch_size must be 1
    #        input vector size must be 1, since it's only a number/index
    # mode: if 'train', feed the y as the inputs
    #       if 'eval', don't feed y and use greedy algorithm
    def forward(self, input_hidden, y=None, mode='train'):
        #assert (mode == 'eval' and y is None) or (mode == 'train' and not y is None), "y shouldn't be None if train mode is set"
        #assert mode == 'eval' or (mode == 'train' and y.shape[0] == 1), "batch size must be 1"
        #assert mode == 'eval' or (mode == 'train' and y.shape[2] == 1), "input vector size must be 1"
        #assert mode == 'eval' or (mode == 'train' and y.shape[1] <= self.max_seq_length), "max length must be respected " \
        #                   + str(y.shape[0]) + " not lower than " + str(self.max_seq_length)

        if y != None:
            batch_size, seq_len, input_vector_size = y.shape
            y = y.reshape(seq_len, batch_size, input_vector_size)

        # this initialization is random, but probably it's a good idea to normalize also the input x,
        # if something doesn't work
        h = input_hidden.view(self.num_layer, 1, self.hidden_size) # this is now given by the encoder!
        c = torch.zeros((self.num_layer, 1, self.hidden_size)) # torch.empty((self.num_layer, batch_size, self.hidden_size)).normal_(mean=0.0, std=0.01)
        h = h.cuda()
        c = c.cuda()

        loss = 0

        output_words = [self.target_SOS_idx]

        if mode == 'train':
            # embedding input(*) - LongTensor of arbitrary shape containing the indices to extract
            # Output: (∗,H), where * is the input shape and H=embedding_dim
            for time_step in range(seq_len-1):
                input = self.embedding(y[time_step, 0, 0]).view(1, 1, -1)
                input = F.relu(input)

                out, (h, c) = self.lstm(input, (h, c))
                # lascio h or c?
                out = out.squeeze(0) # we need to remove one dimension, since we care only about one time step
                out = self.out_fc(out) #
                out = F.relu(out) # <- forse non c'e' l'usanza di mettere una fun di attivazione dopo la softmax..
                out = self.log_softmax(out) # I need to indicate where's the dimension on which I want to softmax
                                            # which is 1 if I have [1,2129]

                topv, topi = out.topk(1)  # ...we predict using greedy strategy from output
                output_words.append(topi.item())
                # print(out.shape) # out has shape [1,2129]
                # print(y[time_step, 0, 0].unsqueeze(0).shape) # y has shape [1]
                # since we are using the Negative Log Likelihood Loss, we don't need to pick the top value by ourselves
                # nlloss require (input:[batch_size, number of classes] e.g.[ 1,2129], target:[N] number of cases )
                loss += loss_criterion(out, y[time_step+1, 0, 0].unsqueeze(0))
        elif mode == 'eval':
            input = torch.tensor(self.target_SOS_idx).cuda()
            # there's no loss when evaluating the decoder since only
            for time_step in range(self.max_seq_length):
                input = self.embedding(input).view(1, 1, -1)
                input = F.relu(input)

                out, (h, c) = self.lstm(input, (h, c))
                out = out.squeeze(0)
                out = self.out_fc(out)
                out = F.relu(out)
                out = self.log_softmax(out)
                #print(out.shape)
                topv, topi = out.topk(1) # ...we predict using greedy strategy from output
                output_words.append(topi.item())
                #print("softmax: " + str(out[:10]) + " topv: " + str(topv.item()) + " topi: " + str(topi.item()))
                #print("vediamo il valore in topi: " + str(out[0][topi.item()]) + ' ' + str(out[0][topi.item()+1]) + ' ' + str(out[0][topi.item()-1]))
                if topi.item() == self.target_EOS_idx:
                    break
                else:
                    input = topi

        return loss, output_words


def train(train_x, train_y, validate_x, validate_y,
          num_epoch, encoder_optimizer, decoder_optimizer, learning_rate,
          num_runs, save_to_file=False, write_summary=False):


    if write_summary:
        writer = SummaryWriter()
        writer.add_scalar('train/learning_rate', learning_rate)

    running_loss = 0

    for epoch in range(num_epoch):
        # shuffle our training set
        num_train_sentences = len(train_x)
        shuffled_train_indexes = random.sample(range(num_train_sentences), num_train_sentences)

        for train_idx in range(num_train_sentences):
            pair_idx = shuffled_train_indexes[train_idx]
            # convert both input and output into vocabulary indexes
            input_vector_x = utils.convert_vector_word2idx(train_x[pair_idx], x_word2idx_dict)
            target_vector_y = utils.convert_vector_word2idx(train_y[pair_idx], y_word2idx_dict)
            # input_vector_x = utils.convert_vector_word2idx(x_sentences[pair_idx], x_word2idx_dict)
            # target_vector_y = utils.convert_vector_word2idx(y_sentences[pair_idx], y_word2idx_dict)
            # print(str(input_vector_x) + ' ' + str(target_vector_y))
            # print(str(x_sentences[pair_idx]) + ' ' + str(y_sentences[pair_idx]), end='----\n')

            input_tensor_x = torch.tensor(input_vector_x).view(-1, 1)  # -> [ seq_len, input_size = 1 since it's just a number ]
            input_tensor_x = input_tensor_x.unsqueeze(0).cuda().type(torch.cuda.LongTensor)  # to create the batch size and length
            target_tensor_y = torch.tensor(target_vector_y).view(-1, 1)
            target_tensor_y = target_tensor_y.unsqueeze(0).cuda().type(torch.cuda.LongTensor)

            _, (last_hidden, _) = encoder(input_tensor_x)

            total_loss, output_words = decoder(last_hidden, y=target_tensor_y)
            # since we need to take into account also the length of the targets we need to divide the total loss
            # with the length of the target sequence,
            # HOWEVER, we notice that the BACKPROPAGATION occurs on the total loss, we do not divide it!
            total_loss.backward()

            # drawing graph

            # writer.add_graph(encoder, (input_tensor_x))
            # writer.add_graph(decoder, (encoder_hidden, target_tensor_y))

            # -------------------------

            running_loss += total_loss.item() / len(target_vector_y)
            iter = epoch * num_train_sentences + train_idx + 1
            if iter % num_runs == 0:
                avg_loss = running_loss / num_runs
                print(str(iter/(num_epoch * num_train_sentences)*100) + " % it: " + str(iter) + " avg loss: " + str(avg_loss))
                running_loss = 0

                # compute norm of all gradients in encoder
                list_of_grad = []
                for module in encoder._modules.values():
                    for params in module._parameters.values():
                        if params.grad is not None:
                            # print(params.grad)
                            # flatten the tensor of weights
                            list_of_grad = list_of_grad + list(torch.flatten(params.grad))
                # print('encoder parameters with gradient length: ' + str(len(list_of_grad)))
                encoder_grad_mean = torch.mean(torch.tensor(list_of_grad))
                # do the same for the decoder
                list_of_grad = []
                for module in encoder._modules.values():
                    for params in module._parameters.values():
                        if params.grad is not None:
                            # print(params.grad)
                            # flatten the tensor of weights
                            list_of_grad = list_of_grad + list(torch.flatten(params.grad))
                # print('decoder parameters with gradient length: ' + str(len(list_of_grad)))
                decoder_grad_mean = torch.mean(torch.tensor(list_of_grad))

                print("Input: " + str(train_x[pair_idx]) + " Gt: " + str(train_y[pair_idx]) + \
                      " Output word: " + str(utils.convert_vector_idx2word(output_words, y_idx2word_list)))

                if write_summary:
                    writer.add_scalar('train/encoder_gradient', encoder_grad_mean, iter)
                    writer.add_scalar('train/decoder_gradient', decoder_grad_mean, iter)
                    writer.add_scalar('train/loss', avg_loss, iter)

                    writer.add_histogram('train_hist_encoder/encoder_embedding_weights', encoder._modules['embedding'].weight, iter)
                    # writer.add_histogram('train_hist_encoder/encoder_out_fc_weights', encoder._modules['out_fc'].weight, iter)
                    # writer.add_histogram('train_hist_encoder/encoder_out_fc_bias', encoder._modules['out_fc'].bias, iter)
                    writer.add_histogram('train_hist_encoder/encoder_lstm_weights_hh_l0', encoder._modules['lstm'].weight_hh_l0, iter)
                    writer.add_histogram('train_hist_encoder/encoder_lstm_bias_hh_l0', encoder._modules['lstm'].bias_hh_l0, iter)
                    writer.add_histogram('train_hist_encoder/encoder_lstm_weights_ih_l0', encoder._modules['lstm'].weight_ih_l0, iter)
                    writer.add_histogram('train_hist_encoder/encoder_lstm_bias_ih_l0', encoder._modules['lstm'].bias_ih_l0, iter)

                    writer.add_histogram('train_hist_decoder/decoder_embedding_weights', decoder._modules['embedding'].weight, iter)
                    writer.add_histogram('train_hist_decoder/decoder_out_fc_weights', decoder._modules['out_fc'].weight, iter)
                    writer.add_histogram('train_hist_decoder/decoder_out_fc_bias', decoder._modules['out_fc'].bias, iter)
                    writer.add_histogram('train_hist_decoder/decoder_lstm_weights_hh_l0', decoder._modules['lstm'].weight_hh_l0, iter)
                    writer.add_histogram('train_hist_decoder/decoder_lstm_bias_hh_l0', decoder._modules['lstm'].bias_hh_l0, iter)
                    writer.add_histogram('train_hist_decoder/decoder_lstm_weights_ih_l0', decoder._modules['lstm'].weight_ih_l0, iter)
                    writer.add_histogram('train_hist_decoder/decoder_lstm_bias_ih_l0', decoder._modules['lstm'].bias_ih_l0, iter)
                    # writer.add_scalar('Accuracy/train', np.random.random(), iter)

            encoder_optimizer.step()
            decoder_optimizer.step()

    if save_to_file:
        torch.save(encoder.state_dict(), './encoder_100k_iter_lr' + str(learning_rate) + '.pth')
        torch.save(decoder.state_dict(), './decoder_100k_iter_lr' + str(learning_rate) + '.pth')
    if write_summary:
        writer.close()

if __name__ == "__main__":

    # Seed setting ---------------------------------------------

    seed = 2343567
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # ----------------------------------------------------------

    print("Machine Translation_RNN: Hello World")

    # 0. Global data
    MAX_SEQUENCE_LENGTH = 10 # maximum length of input or output sequence

    # 1. Loading data and pre-processing
    x_sentences, y_sentences = utils.loadDataset()
    x_sentences = utils.normalize(x_sentences)
    y_sentences = utils.normalize(y_sentences)
    x_sentences, y_sentences = utils.filter_out_prefixes(x_sentences, y_sentences)
    x_sentences, y_sentences = utils.filter_out_length(x_sentences, y_sentences, MAX_SEQUENCE_LENGTH)

    x_word2idx_dict, x_idx2word_list, x_vocabulary_size = utils.compute_vocabulary(x_sentences)
    y_word2idx_dict, y_idx2word_list, y_vocabulary_size = utils.compute_vocabulary(y_sentences)
    print("x vocabulary size: " + str(len(x_idx2word_list)))
    print("y vocabulary size: " + str(len(y_idx2word_list)))

    # split each sentence into word and add SOS and EOS to every sentence in x and y
    x_sentences = utils.tokenize_add_EOS_and_SOS(x_sentences)
    y_sentences = utils.tokenize_add_EOS_and_SOS(y_sentences)

    train_x, train_y, validate_x, validate_y, test_x, test_y \
        = utils.split_train_validation_test(x_sentences, y_sentences, train_perc=0.8, validate_perc=0.05, test_perc=0.15)
    print("train len: " + str(len(train_x)))
    print("validate len: " + str(len(validate_x)))
    print("test len: " + str(len(test_x)))

    # 2. Architecture building
    loss_criterion = nn.NLLLoss()
    embedding_size = 128
    hidden_size = 128
    encoder = EncoderRNN(hidden_size, embedding_size, len(x_idx2word_list), MAX_SEQUENCE_LENGTH)
    encoder.cuda()
    encoder.load_state_dict(torch.load('./encoder_800_epoch_lr1e-06.pth'))
    decoder = DecoderRNN(hidden_size, embedding_size, len(y_idx2word_list), MAX_SEQUENCE_LENGTH, loss_criterion, y_word2idx_dict['SOS'], y_word2idx_dict['EOS'])
    decoder.cuda()
    decoder.load_state_dict(torch.load('./decoder_800_epoch_lr1e-06.pth'))


    # 3. Training
    """
    num_epoch = 200
    # provati e falliti:
    learning_rate = 1e-6 # 1e-3 # 1e-3 nope proprio, 1e-4 nope proprio ancora # 1e-5 40k 2.7 SGD # 1e-6, 75k 2.5 loss SGD, 1e-7 non troppo bene..
    # uso tra l'altro Adam invece di SGD...
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    # statitistics
    num_runs = 10000
    # since we need to take into account also the length of the targets we need to divide the total loss
    # with the length of the target sequence
    
    train(train_x=train_x, train_y=train_y, validate_x=validate_x, validate_y=validate_y,
          num_epoch=num_epoch, encoder_optimizer=encoder_optimizer, decoder_optimizer=decoder_optimizer,
          learning_rate=learning_rate, num_runs=num_runs, save_to_file=False, write_summary=False)
    """

    # 4. Parte di Evaluation

    # This part is quite delicate, each input sentence has several correct references
    #
    # I'm shy.	Io sono timido.
    # I'm shy.	Sono timida.
    # I'm shy.	Io sono timida.
    #

    """
    for idx in range(len(test_x)):
        yyy = utils.find_all_correct_sentences(test_x[idx], test_x, test_y)
        print(test_x[idx])
        print(yyy)
        print('\n\n')
    exit()
    """

    for pair_idx in range(len(test_x)):
        # convert both input and output into vocabulary indexes
        input_vector_x = utils.convert_vector_word2idx(test_x[pair_idx], x_word2idx_dict)
        target_vector_y = utils.convert_vector_word2idx(test_y[pair_idx], y_word2idx_dict)
        #print(str(input_vector_x) + ' ' + str(target_vector_y))
        #print(str(x_sentences[pair_idx]) + ' ' + str(y_sentences[pair_idx]), end='----\n')

        input_tensor_x = torch.tensor(input_vector_x).view(-1, 1) # -> [ seq_len, input_size = 1 since it's just a number ]
        input_tensor_x = input_tensor_x.unsqueeze(0).cuda().type(torch.cuda.LongTensor) # to create the batch size and length
        #target_tensor_y = torch.tensor(target_vector_y).view(-1, 1)
        #target_tensor_y = target_tensor_y.unsqueeze(0).cuda().type(torch.cuda.LongTensor)

        _, (last_hidden, _) = encoder(input_tensor_x)

        _, output_words = decoder(last_hidden, mode='eval')

        pred_sentence = utils.convert_vector_idx2word(output_words, y_idx2word_list)
        print("Input: " + str(test_x[pair_idx]) + " Pred " + \
              str(pred_sentence) + " Gt: " + str(test_y[pair_idx]))

        correct_sentences = utils.find_all_correct_sentences(test_x[pair_idx], test_x, test_y)
        bleu_score = nltk.translate.bleu_score.sentence_bleu(correct_sentences, pred_sentence, weights=(0.5, 0.5))
        print("BLEU Score: " + str(bleu_score))