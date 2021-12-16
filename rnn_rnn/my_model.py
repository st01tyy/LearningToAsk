import torch
from torch import nn
from typing import Optional
from encoder import Encoder
from decoder import Decoder


class MyModel(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int,
                 lstm_hidden_size: int = 600, lstm_dropout: float = 0.3, lstm_num_layers: int = 2,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        self.encoder = Encoder(embedding_dim, vocab_size, lstm_hidden_size, lstm_dropout, lstm_num_layers,
                               pretrained_embeddings)
        self.decoder = Decoder(embedding_dim, vocab_size, lstm_hidden_size, lstm_dropout, lstm_num_layers,
                               pretrained_embeddings)

    def forward(self, inputs: torch.LongTensor, targets: torch.LongTensor):
        """

        :param inputs: 输入token序列，形状为(batch_size, input_length)
        :param targets: 样例输出token ID序列，形状为(batch_size, output_length)
        :return: outputs: 模型输出token ID序列，形状为(batch_size, output_length - 1)
        """
        encoder_outputs, hidden = self.encoder(inputs=inputs)
        t_targets = targets.t()
        outputs = torch.Tensor().cuda()
        for i in range(0, targets.size()[1] - 1):
            temp = t_targets[0:i + 1]
            temp = temp.t()
            decoder_outputs = self.decoder(inputs=temp, encoder_outputs=encoder_outputs, hidden=hidden)
            res = torch.argmax(decoder_outputs, dim=-1)  # res size: (batch_size)
            outputs = torch.cat((outputs, res), dim=0)
        outputs = outputs.t()   # outputs size: (batch_size, output_length - 1)
        return outputs
