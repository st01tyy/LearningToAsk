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
        前向传播
        :param encoder_outputs: 编码器所有时刻的隐含层输出，形状为(batch_size, input_length, 2 * lstm_hidden_size)
        :param inputs: 输入序列，形状为(batch_size, input_length, vocab_size)
        :param hidden: Encoder最后时刻的隐藏层输出h_n和c_n
        :return: Decoder最后时刻，最后一层输出的token分类预测，形状为(batch_size, vocab_size)
        """

        inputs = self.embedding(inputs)
        _, (hn, cn) = self.lstm(inputs, hidden)

        hn = hn[self.lstm.num_layers - 1]  # hn size: (batch_size, lstm_hidden_size)    # 只提取最后一层的hn

        # 计算注意力分数
        score = torch.unsqueeze(input=hn, dim=1)  # score size: (batch_size, 1, lstm_hidden_size)
        score = self.attention_linear(score)  # score size: (batch_size, 1, 2 * lstm_hidden_size)
        # t_encoder_outputs size: (batch_size, 2 * lstm_hidden_size, input_length)
        t_encoder_outputs = torch.transpose(input=encoder_outputs, dim0=1, dim1=2)
        score = score.bmm(t_encoder_outputs)  # score size: (batch_size, 1, input_length)
        score = functional.softmax(input=score, dim=-1)

        # 计算Encoder outputs的注意力分数加权和
        c = score.bmm(encoder_outputs)          # c size: (batch_size, 1, 2 * lstm_hidden_size)
        c = torch.squeeze(input=c, dim=1)  # c size: (batch_size, 2 * lstm_hidden_size)

        output = torch.cat((hn, c), dim=1)  # output size: (batch_size, 3 * lstm_hidden_size)
        # 全连接+tanh
        output = self.full_connection_linear(output)    # output size: (batch_size, vocab_size)
        output = torch.tanh(output)    # output size: (batch_size, vocab_size)
        output = self.output_linear(output)  # output size: (batch_size, vocab_size)
        return functional.softmax(input=output, dim=-1)
