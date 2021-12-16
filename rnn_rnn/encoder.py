import torch
from torch import nn
from typing import Optional


class Encoder(nn.Module):
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
                            batch_first=True, bidirectional=True, num_layers=lstm_num_layers)

    def forward(self, inputs: torch.LongTensor):
        """
        前向转播
        :param inputs: 输入序列，形状为(batch_size, input_length, vocab_size)
        :returns: hn, cn
        :return: h_n: 时刻n的隐含层输出，形状为(lstm_num_layers, batch_size, lstm_hidden_size)
        :return: c_n: 时刻n的记忆细胞，形状为(lstm_num_layers, batch_size, lstm_hidden_size)
        """
        inputs = self.embedding(inputs)
        encoder_outputs, (hn, cn) = self.lstm(inputs)

        # 将两个方向的输出向量相加
        temp = torch.unsqueeze(hn[0] + hn[1], dim=0)
        for i in range(1, self.lstm.num_layers):
            temp = torch.cat((temp, torch.unsqueeze(hn[2 * i] + hn[2 * i + 1], dim=0)), dim=0)
        hn = temp
        temp = torch.unsqueeze(cn[0] + cn[1], dim=0)
        for i in range(1, self.lstm.num_layers):
            temp = torch.cat((temp, torch.unsqueeze(cn[2 * i] + cn[2 * i + 1], dim=0)), dim=0)
        cn = temp

        return encoder_outputs, (hn, cn)
