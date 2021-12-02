from torch import nn
from torch.nn import functional
import torch
from typing import Optional
from typing import Tuple


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int,
                 lstm_hidden_size: int = 600, lstm_dropout: float = 0.3, lstm_num_layers: int = 2,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        """

        :param embedding_dim: 词向量维度
        :param vocab_size: 词表大小
        :param lstm_hidden_size: LSTM模型向量维度
        :param lstm_dropout: LSTM层之间的Dropout概率
        :param lstm_num_layers: LSTM层数
        :param pretrained_embeddings: 预训练词向量，形状为(vocab_size, embedding_dim)
        """
        super().__init__()
        if pretrained_embeddings is None:
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        else:
            self.embedding = nn.Embedding.from_pretrained(pretrained_embeddings)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=lstm_hidden_size, dropout=lstm_dropout,
                            batch_first=True, bidirectional=False, num_layers=lstm_num_layers)
        self.linear = nn.Linear(in_features=lstm_hidden_size, out_features=vocab_size)

    def forward(self, inputs: torch.LongTensor, hidden: Tuple):
        """

        :param inputs: 输入序列，形状为(batch_size, input_length, vocab_size)
        :param hidden: Encoder最后时刻的隐藏层输出h_n和c_n
        :return: Decoder最后时刻输出的token分类预测，形状为(num_layers, batch_size,
        """
        inputs = self.embedding(inputs)
        _, (hn, cn) = self.lstm(inputs, hidden)
        hn = self.linear(hn)
        return functional.log_softmax(input=hn, dim=-1)
