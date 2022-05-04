import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNClassificationModel(nn.Module):
    def __init__(self, num_genre,
                 pretrained_vector_weight=None,
                 use_multi_channel=True):
        super(CNNClassificationModel, self).__init__()

        self.num_genre_emb = 10
        self.input_channel = 1
        self.use_multi_channel = use_multi_channel
        self.num_filter = 100

        self.linear_genre = nn.Linear(num_genre, self.num_genre_emb)

        self.embedding_layer = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_vector_weight),
                                                            freeze=False)
        self.embedding_size = pretrained_vector_weight.shape[1]

        if use_multi_channel:
            self.embedding_layer2 = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_vector_weight),
                                                                 freeze=True)
            self.input_channel = 2

        self.convolution_layer_3dfilter = nn.Conv2d(self.input_channel, self.num_filter, (3, self.embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer_3dfilter.weight)

        self.convolution_layer_4dfilter = nn.Conv2d(self.input_channel, self.num_filter, (4, self.embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer_4dfilter.weight)

        self.convolution_layer_5dfilter = nn.Conv2d(self.input_channel, self.num_filter, (5, self.embedding_size))
        nn.init.xavier_uniform_(self.convolution_layer_4dfilter.weight)

        self.dropout = nn.Dropout(p=0.5)

        if num_genre > 0:
            self.linear = nn.Linear(3*self.num_filter+self.num_genre_emb, 1)
        else:
            self.linear = nn.Linear(3*self.num_filter, 1)

        nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, input_sentence, input_genre=None):

        if input_genre is not None:
            input_genre = self.linear_genre(input_genre)

        embedded = self.embedding_layer(input_sentence)

        if self.use_multi_channel:
            embedded2 = self.embedding_layer2(input_sentence)
            embedded = torch.stack([embedded, embedded2], dim=1)
        else:
            embedded = embedded.unsqueeze(1)

        conv_opt3 = self.convolution_layer_3dfilter(embedded)
        conv_opt4 = self.convolution_layer_4dfilter(embedded)
        conv_opt5 = self.convolution_layer_5dfilter(embedded)

        conv_opt3 = F.relu(conv_opt3).squeeze(3)
        conv_opt4 = F.relu(conv_opt4).squeeze(3)
        conv_opt5 = F.relu(conv_opt5).squeeze(3)

        conv_opt3 = F.max_pool1d(conv_opt3, conv_opt3.size(2)).squeeze(2)
        conv_opt4 = F.max_pool1d(conv_opt4, conv_opt4.size(2)).squeeze(2)
        conv_opt5 = F.max_pool1d(conv_opt5, conv_opt5.size(2)).squeeze(2)

        conv_opt = torch.cat((conv_opt3, conv_opt4, conv_opt5), 1)
        conv_opt = self.dropout(conv_opt)

        if input_genre is not None:
            conv_opt = torch.cat([conv_opt, input_genre], dim=-1)

        linear_opt = self.linear(conv_opt)
        pred = torch.sigmoid(linear_opt)

        return pred
