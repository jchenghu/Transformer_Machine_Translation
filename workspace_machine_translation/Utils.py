import random # creating training, validation, testing set
import re # regexp for replacing easily characters

def loadDataset():

    path = './'
    file_name = 'ita.txt'

    x_sentences = []
    y_sentences = []

    with open(path + file_name, 'r') as f:

        for line in f:
            splitted = line.split('\t')
            to_be_translated = splitted[0]
            translated = splitted[1]
            # print('x: ' + str(to_be_translated) + ' y: ' + str(translated))
            x_sentences.append(to_be_translated)
            y_sentences.append(translated)

    return x_sentences, y_sentences

def normalize(sentences):
    for i in range(len(sentences)):
        # convert everything to lowercase
        sentences[i] = sentences[i].lower()
        # remove [ . , @ ! ? '  and numbers ]
        sentences[i] = re.sub(r"[.,@!?']+", r" ", sentences[i])
    return sentences

def filter_out_prefixes(x_sentences, y_sentences):
    assert x_sentences != None or y_sentences != None or \
            len(x_sentences) != 0 or y_sentences != 0, "invalid x and y"
    assert len(x_sentences) == len(y_sentences), "x and y length mismatch"

    # !!!!!!!!!!!!!!!!!!!!!!!We restrict ourselves to he is, they are and so on, BECAUSE reducing the number of dataset
    # is not enough if we want to crate a simple example, we need also to restrict the possible cases
    # otherwise it will always perform badly !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # filter out sentences that do not start with
    accepted_prefixes = (
        'i am', 'i m',
        'she is', 'she s',
        'he is', 'he s',
        'it is', 'it s',
        'we are', 'we re',
        'you are', 'you re',
        'they are', 'they re'
    )
    accepted_sentence_idxes = [idx for idx in range(len(x_sentences)) if x_sentences[idx].startswith(accepted_prefixes)]
    x_sentences = [x_sentences[idx] for idx in accepted_sentence_idxes]
    y_sentences = [y_sentences[idx] for idx in accepted_sentence_idxes]

    return x_sentences, y_sentences

def filter_out_length(x_sentences, y_sentences, filter_length):
    accepted_sentence_idxes = [idx for idx in range(len(x_sentences)) \
                                if len(x_sentences[idx]) <= filter_length or len(y_sentences[idx]) <= filter_length]
    x_sentences = [x_sentences[idx] for idx in accepted_sentence_idxes]
    y_sentences = [y_sentences[idx] for idx in accepted_sentence_idxes]
    return x_sentences, y_sentences

#def filter_out_characters(x_sentences, y_sentences, list_prohibited_characters):
#    accepted_sentence_idxes = [idx for idx in range(len(x_sentences)) \
#                if not list_prohibited_characters in x_sentences[idx] and not list_prohibited_characters in y_sentences[idx]]
#    x_sentences = [x_sentences[idx] for idx in accepted_sentence_idxes]
#    y_sentences = [y_sentences[idx] for idx in accepted_sentence_idxes]


def split_train_validation_test(x_sentences, y_sentences, train_perc, validate_perc, test_perc, seed = None):
    assert train_perc + validate_perc + test_perc == 1.0, "ratios do not sum up to 1"

    if seed != None:
        random.seed(seed)

    num_senteces = len(x_sentences)
    shuffled_idxes = random.sample(range(num_senteces), num_senteces)
    num_train = int(num_senteces*train_perc)
    num_validate = int(num_senteces*validate_perc)
    num_test = num_senteces - num_train - num_validate

    train_x = [x_sentences[shuffled_idxes[i]] for i in range(num_train)]
    train_y = [y_sentences[shuffled_idxes[i]] for i in range(num_train)]
    validate_x = [x_sentences[shuffled_idxes[i]] for i in range(num_train, num_train + num_validate)]
    validate_y = [y_sentences[shuffled_idxes[i]] for i in range(num_train, num_train + num_validate)]
    test_x = [x_sentences[shuffled_idxes[i]] for i in range(num_train + num_validate, num_train + num_validate + num_test)]
    test_y = [y_sentences[shuffled_idxes[i]] for i in range(num_train + num_validate, num_train + num_validate + num_test)]

    return train_x, train_y, validate_x, validate_y, test_x, test_y

def compute_vocabulary(sentences):
    # we want to find all the unique words in all sentences:
    # 1. first I need to flatten all the list
    #    [['I have a dog'], ['I have a cat']] becomes
    #    [['I','have','a','dog'], ['I','have','a','cat']]
    #    becomes: ['I','have','a','dog','I','have','a','cat']
    # 2. remove repeated words

    flattened_list_of_strings = [ string for sentence in sentences for string in sentence.split(' ')]
    flattened_list_of_strings = set(flattened_list_of_strings)
    if '' in flattened_list_of_strings:
        flattened_list_of_strings.remove('')
    flattened_list_of_strings.add('SOS')
    flattened_list_of_strings.add('EOS')
    flattened_list_of_strings = list(flattened_list_of_strings)

    # Sort the list - this is very important for preserving reproducibility
    flattened_list_of_strings.sort()

    word2idx_dict = dict()
    idx2word_list = list()
    for idx in range(len(flattened_list_of_strings)):
        word2idx_dict[flattened_list_of_strings[idx]] = idx
        idx2word_list.append(flattened_list_of_strings[idx])

    return word2idx_dict, idx2word_list, len(idx2word_list)

def compute_vocabulary_with_PAD(sentences):
    # we want to find all the unique words in all sentences:
    # 1. first I need to flatten all the list
    #    [['I have a dog'], ['I have a cat']] becomes
    #    [['I','have','a','dog'], ['I','have','a','cat']]
    #    becomes: ['I','have','a','dog','I','have','a','cat']
    # 2. remove repeated words

    flattened_list_of_strings = [ string for sentence in sentences for string in sentence.split(' ')]
    flattened_list_of_strings = set(flattened_list_of_strings)
    if '' in flattened_list_of_strings:
        flattened_list_of_strings.remove('')
    flattened_list_of_strings.add('SOS')
    flattened_list_of_strings.add('EOS')
    flattened_list_of_strings.add('PAD') # this if for the transformer
    flattened_list_of_strings = list(flattened_list_of_strings)

    # Sort the list - this is very important for preserving reproducibility
    flattened_list_of_strings.sort()

    word2idx_dict = dict()
    idx2word_list = list()
    for idx in range(len(flattened_list_of_strings)):
        word2idx_dict[flattened_list_of_strings[idx]] = idx
        idx2word_list.append(flattened_list_of_strings[idx])

    return word2idx_dict, idx2word_list, len(idx2word_list)

# for the evaluation phase I need to compute BLEU, which needs all reference sentences
def find_all_correct_sentences(sentence, x_sentences, y_sentences):
    return [ y_sentences[idx] for idx in range(len(x_sentences)) if sentence == x_sentences[idx] ]


def tokenize_add_EOS_and_SOS(sentences):
    for i in range(len(sentences)):
        sentences[i] = ['SOS'] +  sentences[i].split(' ') + ['EOS']
        while '' in sentences[i]: # ce ne potrebbero essere piu' di uno!
            sentences[i].remove('')
    return sentences

# note: it preserves the sorting of the list
def add_PAD_according_to_batch(batch_sentences, word2idx_dict):
    # 1. first find the longest sequence here
    batch_size = len(batch_sentences)
    list_of_lengthes = [ len(batch_sentences[batch_idx]) for batch_idx in range(batch_size)]
    in_batch_max_seq_len = max(list_of_lengthes)
    # 2. add 'PAD' tokens until all the batch have same seq_len
    for batch_idx in range(batch_size):
        batch_sentences[batch_idx] = batch_sentences[batch_idx] \
            + [word2idx_dict['PAD']]*(in_batch_max_seq_len - len(batch_sentences[batch_idx]))
    return batch_sentences

def convert_vector_word2idx(sentence, word2idx_dict):
    sentence = [ word2idx_dict[word] for word in sentence]
    return sentence

def convert_allsentences_word2idx(sentences, word2idx_dict):
    for i in range(len(sentences)):
        sentences[i] = convert_vector_word2idx(sentences[i], word2idx_dict)
    return sentences

def convert_vector_idx2word(sentence, idx2word_list):
    sentence = [ idx2word_list[idx] for idx in sentence]
    return sentence

# def compute_bleu_score(correct_sentences, target_sentences):


if __name__ == "__main__":

    x_sentences, y_sentences = loadDataset()
    print('dataset size before filters: ' + str(len(x_sentences)))
    x_sentences = normalize(x_sentences)
    y_sentences = normalize(y_sentences)
    x_sentences, y_sentences = filter_out_prefixes(x_sentences, y_sentences)
    print('dataset after filtering \'he is, they are ecc...\': ' + str(len(x_sentences)))
    x_sentences, y_sentences = filter_out_length(x_sentences, y_sentences, 10)
    print('dataset after filtering length: ' + str(len(x_sentences)))

    x_word2idx_dict, x_idx2word_list, x_vocabulary_size = compute_vocabulary(x_sentences)
    y_word2idx_dict, y_idx2word_list, y_vocabulary_size = compute_vocabulary(y_sentences)
    print("x vocabulary size: " + str(len(x_idx2word_list)) + ' ' + str(x_idx2word_list))
    print("y vocabulary size: " + str(len(y_idx2word_list)) + ' ' + str(y_idx2word_list))

    # split each sentence into word and add SOS and EOS to every sentence in x and y
    x_sentences = tokenize_add_EOS_and_SOS(x_sentences)
    y_sentences = tokenize_add_EOS_and_SOS(y_sentences)

    train_x, train_y, validate_x, validate_y, test_x, test_y \
        = split_train_validation_test(x_sentences, y_sentences, train_perc=0.8, validate_perc=0.05, test_perc=0.15)
    print("train len: " + str(len(train_x)))
    print("validate len: " + str(len(validate_x)))
    print("test len: " + str(len(test_x)))
