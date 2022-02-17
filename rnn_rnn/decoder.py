import torch
from torch import nn
from torch.nn import functional
from typing import Optional, Tuple


class Decoder(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int,
                 lstm_hidden_size: int = 600, lstm_dropout: float = 0.3, lstm_num_layers: int = 2,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        """
        构造函数
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
        self.attention_linear = nn.Linear(in_features=lstm_hidden_size, out_features=2 * lstm_hidden_size)
        self.full_connection_linear = nn.Linear(in_features=3 * lstm_hidden_size, out_features=vocab_size)
        self.output_linear = nn.Linear(in_features=vocab_size, out_features=vocab_size)

    def forward(self, inputs: torch.LongTensor, hidden: Tuple, encoder_outputs: torch.Tensor):
        """

        :param encoder_outputs: (batch_size, encoder_length, 2 * lstm_hidden_size)
        :param inputs: (batch_size, decoder_length)
        :param hidden: Decoder上一时刻的隐含层输出或Encoder最后时刻的隐藏层输出
        :return: Decoder每一时刻的单词分类，形状为(batch_size, decoder_length, vocab_size)
        """

        inputs = self.embedding(inputs)
        hidden_outputs, _= self.lstm(inputs, hidden)    #hidden_outputs: (batch_size, decoder_length, lstm_hidden_size)

        # 计算注意力分数
        score = self.attention_linear(hidden_outputs)  # score size: (batch_size, decoder_length, 2 * lstm_hidden_size)
        # t_encoder_outputs size: (batch_size, 2 * lstm_hidden_size, encoder_length)
        t_encoder_outputs = torch.transpose(input=encoder_outputs, dim0=1, dim1=2)
        score = score.bmm(t_encoder_outputs)  # score size: (batch_size, decoder_length, encoder_length)
        score = functional.softmax(input=score, dim=-1)

        # 计算Encoder outputs的注意力分数加权和
        c = score.bmm(encoder_outputs)          # c size: (batch_size, decoder_length, 2 * lstm_hidden_size)

        output = torch.cat((hidden_outputs, c), dim=2)  # output size: (batch_size, decoder_length, 3 * lstm_hidden_size)
        # 全连接+tanh
        output = self.full_connection_linear(output)    # output size: (batch_size, decoder_length, vocab_size)
        output = torch.tanh(output)    # output size: (batch_size, decoder_length, vocab_size)
        output = self.output_linear(output)  # output size: (batch_size, decoder_length, vocab_size)
        return output
