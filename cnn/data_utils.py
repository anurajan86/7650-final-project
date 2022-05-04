import ast

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split


def generate_vocab(cols, vocab_size, vocab, word_to_id, id_to_word, full_df):
    for index, row in full_df.iterrows():

        for c in cols:

            s2n = []
            split = row[c].split()

            for word in split:

                if word not in vocab:
                    vocab.add(word)
                    word_to_id[word] = vocab_size
                    # word_to_count[word] = 1
                    s2n.append(vocab_size)
                    id_to_word[vocab_size] = word
                    vocab_size += 1
                else:
                    # word_to_count[word] += 1
                    s2n.append(word_to_id[word])

            full_df.at[index, c] = s2n

    return full_df, vocab, word_to_id, id_to_word


def split_data(full_df):
    train_ratio = 0.8  # Split into training and validation
    test_ratio = 0.85  # Split into training and testing

    cols = ['cleaned_reviews'] + list(full_df.columns[3:])
    X = list(full_df[cols].iterrows())
    y = list(full_df['is_spoiler'].astype(int))

    X_rem, X_test, y_rem, y_test = train_test_split(X, y, train_size=test_ratio, shuffle=True, random_state=0)
    X_train, X_val, y_train, y_val = train_test_split(X_rem, y_rem, train_size=train_ratio, shuffle=True, random_state=0)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def pad_sequences(X_list, genre_flag=False):
    if not genre_flag:
        X_padded = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(l) for l in X_list]).type(torch.LongTensor).t()  # padding the sequences with 0
        X_mask = torch.nn.utils.rnn.pad_sequence([torch.as_tensor([1.0] * len(l)) for l in X_list]).type(torch.FloatTensor).t()
        X_genre = None
    else:
        X_padded = torch.nn.utils.rnn.pad_sequence([torch.as_tensor(l[1][0]) for l in X_list]).type(torch.LongTensor).t()  # padding the sequences with 0
        X_mask = torch.nn.utils.rnn.pad_sequence([torch.as_tensor([1.0] * len(l[1][0])) for l in X_list]).type(torch.FloatTensor).t()
        X_genre = torch.stack([torch.as_tensor(l[1][1:]).type(torch.FloatTensor) for l in X_list])
    return X_padded, X_mask, X_genre


def read_GloVe(filename):
    embeddings = {}
    for line in open(filename).readlines():
        # print(line)
        fields = line.strip().split(" ")
        word = fields[0]
        embeddings[word] = [float(x) for x in fields[1:]]

    return embeddings


def init_glove(GloVe, word_to_id, emb_dim):
    pretrained = torch.zeros(max(word_to_id.values()) + 1, emb_dim)
    scale = torch.sqrt(torch.Tensor([3.0 / emb_dim]))[0]
    for word, i in word_to_id.items():
        if word in GloVe:
            pretrained[i] = torch.FloatTensor(GloVe[word])
        else:
            vect = torch.FloatTensor(emb_dim).uniform_(-scale, scale)
            pretrained[i] = torch.FloatTensor(vect)

    return pretrained


def add_genre_cols(df):
    genre_list = list(set([x_ for x in df.genre for x_ in ast.literal_eval(x)]))
    row_array = []
    for x in df.genre:
        row = np.zeros(len(genre_list), dtype=bool)
        row[[genre_list.index(x_) for x_ in ast.literal_eval(x)]] = True
        row_array.append(row)
    row_array = np.stack(row_array)
    return df.join(pd.DataFrame(row_array, columns=genre_list)), genre_list


def prepare_data(genre_flag):
    emb_dim = 300
    vocab = set()
    word_to_id = dict()
    id_to_word = dict()
    word_to_count = dict()
    vocab_size = 1
    cols = ['is_spoiler', 'cleaned_reviews']
    genre_list = []
    if genre_flag:
        cols += ['genre']

    full_df = pd.read_csv("cleaned_reviews_summaries.zip", usecols=cols)
    full_df, vocab, word_to_id, id_to_word = generate_vocab([cols[1]], vocab_size, vocab, word_to_id, id_to_word, full_df)
    full_df['is_spoiler'] = full_df['is_spoiler'].fillna(0)

    if genre_flag:
        full_df['genre'] = full_df['genre'].fillna('[]')
        full_df, genre_list = add_genre_cols(full_df)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(full_df)

    X_train = X_train[:int(len(X_train) * 0.25)]
    y_train = y_train[:int(len(y_train) * 0.25)]

    # genre_spoiler_ratios = [full_df[full_df.iloc[:, 3 + i]].is_spoiler.sum() / full_df.iloc[:, 3 + i].sum() for i in
    #                         range(len(genre_list))]
    # idx = np.argsort(genre_spoiler_ratios)
    # plt.figure(figsize=(10, 8))
    # plt.bar([genre_list[i] for i in idx], [genre_spoiler_ratios[i] for i in idx])
    # plt.xticks(rotation=45)
    # plt.title('Spoiler ratio by genre')
    # plt.show()

    data = (X_train, y_train), (X_val, y_val), (X_test, y_test)

    GloVe = read_GloVe("glove.6B.300d.txt")
    pretrained = init_glove(GloVe, word_to_id, emb_dim)

    print(genre_list)

    return data, pretrained, vocab, word_to_id, id_to_word, genre_list
