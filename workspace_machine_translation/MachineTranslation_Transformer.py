import random
import numpy as np
import torch
import math
import copy
from time import time

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

import Utils as utils

# modules for word processing
class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(Embedder, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoder, self).__init__()
        assert d_model % 2 == 0, "d_model is not even, even number suggested"

        self.d_model = d_model

        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / 10000 ** ((2 * i)/d_model))
                pe[pos, i+1] = math.cos(pos / 10000 ** ((2 * (i + 1)) / d_model))
        pe = pe.unsqueeze(0)

        # Questo e' solo un modo cool per mettere in self.pe la variabile appena creata
        # ma che non deve essere un qualcosa di trainabile
        self.register_buffer('pe', pe)

    # x shape [ batch_size, seq_len, d_model]
    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.shape[1]
        # we apply this to each row of the batch, fortunately it's automatically broadcasted all along batches
        # thanks to the pe = pe.unsqueeze(0)
        x = x + self.pe[0, :seq_len]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model):
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model

        # Per ora non lo faccio Multi-Head per non mettere troppe cose nell'implementazione
        # in sostanza vorrei avere un "hidden_size" di d_model, ma in pratica lo divido nel numero di heads
        # self.d_k = d_model # // num_heads # '//' vuol dire, dividi e tronca al intero inferiore
        # self.num_heads = num_heads

        self.Q_linear = nn.Linear(d_model, d_model)
        self.K_linear = nn.Linear(d_model, d_model)
        self.V_linear = nn.Linear(d_model, d_model)

        self.out_linear = nn.Linear(d_model, d_model)

    # q,k,v shape: [batch_size, seq_len, vector_size]
    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, _ = q.shape

        k = self.K_linear(k).view(batch_size, -1, self.d_model)
        q = self.Q_linear(q).view(batch_size, -1, self.d_model)
        v = self.V_linear(v).view(batch_size, -1, self.d_model)
        # outputhere: [ batch_size, seq_len, d_model ]

        # torch.matmul( [ batch_size, seq_len, d_model ], [ batch_size, d_model, seq_len ]
        # outputhere: [ batch_size, seq_len, seq_len ]
        sim_scores = torch.matmul(q, k.transpose(2, 1))
        sim_scores = sim_scores / (self.d_model ** 0.5) # scaling by sqrt of d_k

        if mask is not None:
            # if there's a mask it must be multiplied with the similarity score so the padding part is ignored
            # or the network can't peak in the next layer, the value is an approximation of negative Inf, since
            # in the softmax will become zero

            # the zero is not applied AFTER the softmax because it would ruin the "probability" property (it would not
            # sum up to zero
            sim_scores = sim_scores.masked_fill(mask == 0, value=-1e9) # sugg: - float('Inf') / -np.inf
                                                                       # warning: exception when all - Inf on softmax
                                                                       # workaround Federico Landi on github
                    # https://github.com/aimagelab/perceive-transform-and-act/blob/master/tasks/R2R/Models/transformer.py
                    # row 90
        # here we can specify the dimension over which softmax is applied
        sim_scores = F.softmax(sim_scores, dim=2)
        # output: [ batch_size, seq_len, seq_len ]

        attention_applied = torch.matmul(sim_scores, v)
        # output: [ batch_size, seq_len, d_model ]

        out = self.out_linear(attention_applied)
        return out

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout_perc=0.1):
        super(FeedForward, self).__init__()
        # dopo la concatenation delle d_k diventa di nuovo d_model
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout_perc)
        self.linear_2 = nn.Linear(d_ff, d_model)

    # x : [ batch_size, seq_len, vector_size = d_model]
    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        # output here: [ batch_size, seq_len, d_model ]
        return x

# ps: usa Layer Norm di pytorch --- different implementation
class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(Norm, self).__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps

    # x shape: [batch_size, seq_len, d_model]
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout_perc=0.1):
        super(EncoderLayer, self).__init__()
        self.norm_1 = Norm(d_model) # one after the multihead
        self.norm_2 = Norm(d_model) # another one after the feed forward
        self.multi_head_attention = MultiHeadAttention(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)

    # we only need one input x since it's repeated over q,k,v
    def forward(self, x, mask):
        # "The encoder is composed of a stack of N = 6 identical layers. Each layer has two
        # sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-
        # wise fully connected feed-forward network. We employ a residual connection [11] around each of
        # the two sub-layers, followed by layer normalization" same for encoder
        x = x + self.dropout_1(self.multi_head_attention(q=x, k=x, v=x, mask=mask))
        x = self.norm_1(x)
        # note the residual connection
        x = x + self.dropout_2(self.ff(x))
        x = self.norm_2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout_perc=0.1):
        super(DecoderLayer, self).__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout_perc)
        self.dropout_2 = nn.Dropout(dropout_perc)
        self.dropout_3 = nn.Dropout(dropout_perc)

        self.multi_head_attention_1 = MultiHeadAttention(d_model)
        self.multi_head_attention_2 = MultiHeadAttention(d_model)

        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x, cross_connection_x, selfattn_mask, crossattn_mask):
        # self attention
        x = x + self.dropout_1(self.multi_head_attention_1(x, x, x, mask=selfattn_mask))
        x = self.norm_1(x)
        # cross_attention
        x = x + self.dropout_2(self.multi_head_attention_2(
            q=x, k=cross_connection_x, v=cross_connection_x, mask=crossattn_mask))
        x = self.norm_2(x)
        x = x + self.dropout_3(self.ff(x))
        x = self.norm_3(x)
        return x

class Transformer(nn.Module):
    def __init__(self, d_model, N, input_word2idx, output_word2idx, max_seq_len, d_ff, dropout_perc=0.1):
        super(Transformer, self).__init__()
        self.N = N
        self.input_word2idx = input_word2idx
        self.output_word2idx = output_word2idx
        self.max_seq_len = max_seq_len

        #self.encoders = [] --> ERRATO!
        #self.decoders = []
        #for i in range(N):
        #    self.encoders.append(EncoderLayer(d_model, d_ff, dropout_perc).cuda())
        #    self.decoders.append(DecoderLayer(d_model, d_ff, dropout_perc).cuda())

        self.encoders = nn.ModuleList([EncoderLayer(d_model, d_ff, dropout_perc) for i in range(N)])
        self.decoders = nn.ModuleList([DecoderLayer(d_model, d_ff, dropout_perc) for i in range(N)])

        self.linear = torch.nn.Linear(d_model, len(output_word2idx))
        # since output has shape [batch_size, seq_len, y_vocabulary] we want softmax to be computed along dim=2,
        # the vocabulary
        self.log_softmax = nn.LogSoftmax(dim=2)

        self.input_embedder = Embedder(len(input_word2idx), d_model)
        self.output_embedder = Embedder(len(output_word2idx), d_model)
        self.positional_encoder = PositionalEncoder(d_model, max_seq_len)

        # initialize all parameters

        for p in self.parameters():
            if p.dim() > 1:
                #nn.init.xavier_uniform_(p)
                nn.init.xavier_normal_(p)


    # enc_x shape [batch_size, max_seq_len]
    # target_sentence [batch_size, max_seq_len]
    # mode = 'train' or 'eval'
    def forward(self, enc_x, dec_x=None, mode='train'):

        pad_mask = create_pad_mask(enc_x.shape[1], enc_x, self.input_word2idx)
        x = self.input_embedder(enc_x)
        x = self.positional_encoder(x) # x shape [batch_size, max_seq_len, d_model]
        for i in range(self.N):
            x = self.encoders[i](x=x, mask=pad_mask)

        # this step is not easy, but the only thing we have to consider is the mask which changes
        # at each time step to show only portion of the decoder input each time.
        if mode == 'train':
            nopeak_and_pad_mask = create_nopeak_and_pad_mask(dec_x, self.output_word2idx)
            # we look at enc_x sequence since the attention matrix which this mask will be applied to has shape
            # [ bs, target_seq_len, source_seq_len ]
            pad_mask = create_pad_mask(dec_x.shape[1], enc_x, self.output_word2idx)
            y = self.output_embedder(dec_x)
            y = self.positional_encoder(y)
            for i in range(self.N):
                y = self.decoders[i](x=y, cross_connection_x=x, selfattn_mask=nopeak_and_pad_mask, crossattn_mask=pad_mask)
            y = self.linear(y)
            y = self.log_softmax(y)
            # just like any seq2seq, during training, the

        elif mode == 'eval':
            output_words = [y_word2idx_dict['SOS']]
            dec_input = [y_word2idx_dict['SOS']]  # this is also the model output since it grows step by step
            for pos in range(self.max_seq_len):
                # at every iteration, the enc_x is the same, but dec_x is updated

                dec_input_tensor = torch.tensor(dec_input).unsqueeze(0).cuda().type(torch.cuda.LongTensor) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                nopeak_and_pad_mask = create_nopeak_and_pad_mask(dec_input_tensor, self.output_word2idx)
                # we look at enc_x sequence since the attention matrix which this mask will be applied to has shape
                # [ bs, target_seq_len, source_seq_len ]
                pad_mask = create_pad_mask(dec_input_tensor.shape[1], enc_x, self.output_word2idx)
                y = self.output_embedder(dec_input_tensor)
                y = self.positional_encoder(y) # [1, 1, 128]
                for i in range(self.N):
                    y = self.decoders[i](x=y, cross_connection_x=x, selfattn_mask=nopeak_and_pad_mask, crossattn_mask=pad_mask)
                y = self.linear(y)
                y = self.log_softmax(y)

                topv, topi = y[0, pos].topk(1)
                output_words.append(topi.item())
                dec_input.append(topi.item())
                if topi.item() == y_word2idx_dict['EOS'] or pos == self.max_seq_len-1:
                    y = output_words
                    break

        return y

# create pad mask request x [ bs, seq_len, d_model ] -> mask shape [ bs, num_rows, seq_len ]
def create_pad_mask(num_rows, x, language_word2idx):
    batch_size, seq_len = x.shape
    # x input [ batch_size, seq_len ]

    # create first mask to remove influence of PADs in attention matrix
    mask = torch.zeros((batch_size, num_rows, seq_len)).cuda().type(torch.cuda.ByteTensor)
    for batch_idx in range(batch_size):
        vector_of_pads = torch.empty((seq_len)).cuda() * language_word2idx['PAD']
        # torch.tensor([False False True]).sum() -> 1
        how_many_pads = (x[batch_idx] == vector_of_pads).sum()
        # pad_mask [ 1 1 0 ]
        #          [ 1 1 0 ]
        #          [ 1 1 0 ]
        mask[batch_idx, :, :] = 1
        mask[batch_idx, :, seq_len - how_many_pads:] = 0

    # output [ batch_size, seq_len, max_seq_len ]
    return mask

# create_no_peak_and_pad_mask create given x shape [bs, seq_len, d_model] -> mask shape [bs, seq_len, seq_len] for
# the self attention modules, however if we use cross-attention refer to create_pad_mask
def create_nopeak_and_pad_mask(x, language_word2idx):

    batch_size, seq_len = x.shape
    # x input [ batch_size, seq_len ]

    # create first mask to remove influence of PADs in attention matrix
    mask = torch.zeros((batch_size, seq_len, seq_len)).cuda().type(torch.cuda.ByteTensor)
    for batch_idx in range(batch_size):
        vector_of_pads = torch.empty((seq_len)).cuda() * language_word2idx['PAD']
        # torch.tensor([False False True]).sum() -> 1
        how_many_pads = (x[batch_idx] == vector_of_pads).sum()
        pad_mask = torch.ones((seq_len, seq_len)).cuda().type(torch.cuda.ByteTensor)
        # pad_mask [ 1 1 0 ]   nopeak_mask [ 1 0 0 ]
        #          [ 1 1 0 ]               [ 1 1 0 ]
        #          [ 1 1 0 ]               [ 1 1 1 ]
        # pad_mask[seq_len-how_many_pads:, :] = 0
        pad_mask[:, seq_len - how_many_pads:] = 0

        nopeak_mask = torch.tril(torch.ones((seq_len, seq_len)).cuda().type(torch.cuda.ByteTensor), diagonal=0)
        mask[batch_idx] = pad_mask & nopeak_mask

    # output [ batch_size, seq_len, max_seq_len ]
    return mask


def train(train_x, train_y, validate_x, validate_y,
          num_epoch, model,
          batch_size, num_runs,
          optimizer, loss_criterion,
          x_word2idx_dict, y_word2idx_dict):

    running_loss = 0
    start_time = time()

    for epoch in range(num_epoch):
        # shuffle our training set
        num_train_sentences = len(train_x)
        shuffled_train_indexes = random.sample(range(num_train_sentences), num_train_sentences)

        num_batch_iter = math.ceil(num_train_sentences / batch_size)

        for batch_idx in range(num_batch_iter):

            model.train()

            batch_indexes = shuffled_train_indexes[batch_idx * batch_size:(batch_idx + 1) * batch_size]

            # ...this may be computationally expensive actually, it can be optimized
            batch_input_x_list = []
            batch_target_y_list = []
            for idx in batch_indexes:
                batch_input_x_list.append(train_x[idx])
                batch_target_y_list.append(train_y[idx])
            batch_input_x = utils.add_PAD_according_to_batch(batch_input_x_list, word2idx_dict=x_word2idx_dict)
            batch_target_y = utils.add_PAD_according_to_batch(batch_target_y_list, word2idx_dict=y_word2idx_dict)

            batch_input_x = torch.tensor(batch_input_x).cuda() # shape [ batch_size, src_seq_len]
            batch_target_y = torch.tensor(batch_target_y).cuda() # shape [ batch_size, trg_seq_len]

            # the decoder input has shape [batch_size, trg_seq_len - 1] because the last one eventually is <EOS>
            pred = model(enc_x=batch_input_x, dec_x=batch_target_y[:, :-1], mode='train')

            total_loss = 0

            # now when computing the loss we need the shifted target_batch that is batch_target_y[:,1:]
            trg_seq_len = batch_target_y.size(1)
            for i in range(len(batch_indexes)):
                single_sentence_loss = 0
                # the loss must account only for the NO-PADS part
                non_pads_words = 0
                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! usa direttamente loss con ignore idx per i pads
                for pos_idx in range(trg_seq_len-1):
                    non_pads_words = non_pads_words + 1
                    #print(str(pos_idx) + ") TRAIN pred: " + str(pred[i, pos_idx]) + " gt: " + str(batch_target_y[i, pos_idx+1]))
                    single_sentence_loss += loss_criterion(pred[i, pos_idx].unsqueeze(0), batch_target_y[i, pos_idx+1].unsqueeze(0))
                    if batch_target_y[i, pos_idx+1] == y_word2idx_dict['EOS']:
                        break

                total_loss += single_sentence_loss / non_pads_words

            # La media e' necessaria, infatti le formule di cross entropy, fanno la media lyngo il parametro
            # della N, che è il batch size, vedi opzione size_average default a True
            # https://pytorch.org/docs/stable/nn.functional.html
            total_loss = total_loss / batch_size

            total_loss.backward()

            # drawing graph

            # writer.add_graph(encoder, (input_tensor_x))
            # writer.add_graph(decoder, (encoder_hidden, target_tensor_y))

            optimizer.step()

            # -------------------------

            running_loss += total_loss.item()
            iter = epoch * num_batch_iter + batch_idx + 1
            if iter % num_runs == 0:

                avg_loss = running_loss / num_runs
                print(str(round(iter / (num_epoch * num_batch_iter) * 100, 3)) + " % it: " + str(iter) + " avg loss: " + str(
                    round(avg_loss, 3)) + ' elapsed time: ' + str(round( (time()-start_time)/60, 3)) + " minutes")
                running_loss = 0

                # ! Warning: "ValueError: expected sequence of length 4 at dim 1 (got 5)" the error is actually
                # there are list with different size so we can't force to be tensor
                #evaluate(validate_x, validate_y, model, y_word2idx_dict, y_idx2word_list, x_idx2word_list)
                #evaluate(batch_input_x_list[:2], batch_target_y_list[:2], model, y_word2idx_dict, y_idx2word_list, x_idx2word_list)

                if write_summary:

                    """
                    if iter % (num_runs * 30) == 0:
                        list_of_grad = []
                        for param in model.parameters():
                            if param.grad is not None:
                                list_of_grad = list_of_grad + list(torch.flatten(param.grad))
                        print("Quanti parametri ho?: " + str(len(list_of_grad)))
                        all_grads = torch.tensor(list_of_grad)
                        grad_mean = torch.mean(all_grads)
                        writer.add_scalar('train/gradient', grad_mean, iter)
                        #writer.add_histogram('train_hist/transformer_weights', , iter)
                    """

                    writer.add_scalar('train/loss', avg_loss, iter)

                    """
                    writer.add_histogram('train_hist_encoder/encoder_embedding_weights',
                                         transformer._modules['embedding'].weight, iter)
                    # writer.add_histogram('train_hist_encoder/encoder_out_fc_weights', encoder._modules['out_fc'].weight, iter)
                    # writer.add_histogram('train_hist_encoder/encoder_out_fc_bias', encoder._modules['out_fc'].bias, iter)
                    writer.add_histogram('train_hist_encoder/encoder_lstm_weights_hh_l0',
                                         transformer._modules['lstm'].weight_hh_l0, iter)
                    writer.add_histogram('train_hist_encoder/encoder_lstm_bias_hh_l0',
                                         transformer._modules['lstm'].bias_hh_l0, iter)
                    writer.add_histogram('train_hist_encoder/encoder_lstm_weights_ih_l0',
                                         transformer._modules['lstm'].weight_ih_l0, iter)
                    writer.add_histogram('train_hist_encoder/encoder_lstm_bias_ih_l0',
                                         transformer._modules['lstm'].bias_ih_l0, iter)
                    """




def evaluate(validate_x, validate_y, model, y_word2idx_dict, y_idx2word_list, x_idx2word_list):

    model.eval()

    for i in range(len(validate_x)):
        enc_x = torch.tensor(validate_x[i]).unsqueeze(0).cuda().type(torch.cuda.LongTensor)
        output_words = model(enc_x, mode='eval')

        print(output_words)

        pred_sentence = utils.convert_vector_idx2word(output_words, y_idx2word_list)
        input_string = utils.convert_vector_idx2word(validate_x[i], x_idx2word_list)
        target_string = utils.convert_vector_idx2word(validate_y[i], y_idx2word_list)
        print("Input: " + str(input_string) + " Pred " + \
              str(pred_sentence) + " Gt: " + str(target_string))

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

    print("Machine Translation_Transformer: Hello World")

    # 0. Global data
    MAX_SEQUENCE_LENGTH = 10  # maximum length of input or output sequence

    # 1. Loading data and pre-processing
    x_sentences, y_sentences = utils.loadDataset()
    x_sentences = utils.normalize(x_sentences)
    y_sentences = utils.normalize(y_sentences)
    x_sentences, y_sentences = utils.filter_out_prefixes(x_sentences, y_sentences)
    x_sentences, y_sentences = utils.filter_out_length(x_sentences, y_sentences, MAX_SEQUENCE_LENGTH)

    x_word2idx_dict, x_idx2word_list, x_vocabulary_size = utils.compute_vocabulary_with_PAD(x_sentences)
    y_word2idx_dict, y_idx2word_list, y_vocabulary_size = utils.compute_vocabulary_with_PAD(y_sentences)
    print("x vocabulary size: " + str(len(x_idx2word_list)))
    print("y vocabulary size: " + str(len(y_idx2word_list)))

    # split each sentence into word and add SOS and EOS to every sentence in x and y
    x_sentences = utils.tokenize_add_EOS_and_SOS(x_sentences)
    y_sentences = utils.tokenize_add_EOS_and_SOS(y_sentences)

    x_sentences = utils.convert_allsentences_word2idx(x_sentences, x_word2idx_dict)
    y_sentences = utils.convert_allsentences_word2idx(y_sentences, y_word2idx_dict)

    train_x, train_y, validate_x, validate_y, test_x, test_y \
        = utils.split_train_validation_test(x_sentences, y_sentences, train_perc=0.8, validate_perc=0.05,
                                            test_perc=0.15)

    print("train len: " + str(len(train_x)))
    print("validate len: " + str(len(validate_x)))
    print("test len: " + str(len(test_x)))

    # 2. Architecture building
    d_model = 128
    N = 1       # paper N 6
    d_ff = 2048
    transformer = Transformer(d_model=d_model, N=N, \
                              input_word2idx=x_word2idx_dict, output_word2idx=y_word2idx_dict, \
                              d_ff=d_ff, max_seq_len=MAX_SEQUENCE_LENGTH)
    transformer.cuda()

    learning_rate = 7e-5
    transformer.load_state_dict(torch.load('./transformer_' + str(MAX_SEQUENCE_LENGTH) + '_lr' + str(learning_rate) + '.pth'))
    # 3. Training parameters
    loss_criterion = nn.NLLLoss()
    num_epoch = 120
    #optimizer = optim.Adam(transformer.parameters(), lr=learning_rate)
    optimizer = optim.SGD(transformer.parameters(), lr=learning_rate)

    # statitistics
    num_runs = 96
    batch_size = 96

    save_to_file = True
    write_summary = False #True
    if write_summary:
        writer = SummaryWriter()
        writer.add_scalar('train/learning_rate', learning_rate)

    train(train_x=train_x, train_y=train_y, validate_x=validate_x, validate_y=validate_y,
          num_epoch=num_epoch, model=transformer,
          batch_size=batch_size, num_runs=num_runs,
          optimizer=optimizer, loss_criterion=loss_criterion,
          x_word2idx_dict=x_word2idx_dict, y_word2idx_dict=y_word2idx_dict)

    if save_to_file:
        torch.save(transformer.state_dict(), './transformer_' + str(MAX_SEQUENCE_LENGTH) + '_lr' + str(learning_rate) + '.pth')
    if write_summary:
        writer.close()


    # 4. Parte di Evaluation

    #evaluate(train_x, train_y, transformer, y_word2idx_dict, y_idx2word_list, x_idx2word_list)
    #evaluate(test_x, test_y, transformer, y_word2idx_dict, y_idx2word_list, x_idx2word_list)

    #my_sentences = [['i', 'am', 'sick'], ['i', 'am', 'fat']]
    #my_translation = [['sono', 'malato'], ['sono', 'grasso']]

    #my_sentences = [['i', 'am']]
    #my_translation = [['io', 'sono']]

    #my_sentences = utils.convert_vector_word2idx(my_sentences, x_word2idx_dict)
    #my_translation = utils.convert_vector_word2idx(my_translation, y_word2idx_dict)

    #evaluate(my_sentences, my_translation, transformer, y_word2idx_dict, y_idx2word_list, x_idx2word_list)