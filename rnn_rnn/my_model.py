import torch
from torch import nn
from torch.nn import functional
from typing import Optional
from encoder import Encoder
from decoder import Decoder


def _check_eos(ids: torch.LongTensor, eos_id: int):
    """

    :param ids: (batch_size * beam_size)
    :param eos_id: int value
    :return: eos_indices
    """
    n = ids.shape[0]
    eos_indices = []
    for i in range(0, n):
        if ids[i] == eos_id:
            eos_indices.append(i)
    return eos_indices


class MyModel(nn.Module):
    def __init__(self, embedding_dim: int, vocab_size: int,
                 lstm_hidden_size: int = 600, lstm_dropout: float = 0.3, lstm_num_layers: int = 2,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = Encoder(embedding_dim, vocab_size, lstm_hidden_size, lstm_dropout, lstm_num_layers,
                               pretrained_embeddings)
        self.decoder = Decoder(embedding_dim, vocab_size, lstm_hidden_size, lstm_dropout, lstm_num_layers,
                               pretrained_embeddings)

    def forward(self,  encoder_inputs: torch.LongTensor, decoder_inputs: torch.LongTensor = None, **beam_search_args):
        """

        :param encoder_inputs: (batch_size, encoder_length)
        :param decoder_inputs: (batch_size, decoder_length)
        :return: outputs: (batch_size, output_length)
        """
        encoder_outputs, hidden = self.encoder(inputs=encoder_inputs)
        if self.training:
            if decoder_inputs is None:
                raise Exception('decoder inputs can\'t be None in training mode')
            return self.decoder(inputs=decoder_inputs, hidden=hidden, encoder_outputs=encoder_outputs)

        else:
            beam_size = int(beam_search_args['beam_size'])
            assert beam_size >= 1
            sos_id = int(beam_search_args['sos_id'])
            eos_id = int(beam_search_args['eos_id'])
            pad_id = int(beam_search_args['pad_id'])
            batch_size = encoder_inputs.shape[0]
            max_length = int(beam_search_args['max_length'])

            is_done = []
            res = []
            ok = 0
            for i in range(0, batch_size):
                is_done.append(False)
                res.append([])

            decoder_inputs = torch.full((batch_size, 1), sos_id, device=encoder_inputs.device)
            outputs = self.decoder(inputs=decoder_inputs, hidden=hidden, encoder_outputs=encoder_outputs)   # (batch_size, 1, vocab_size)
            outputs = torch.squeeze(outputs, dim=1) # (batch_size, vocab_size)
            outputs = functional.log_softmax(outputs, dim=1)
            scores, ids = torch.topk(outputs, beam_size, dim=1, largest=True)   # (batch_size, beam_size)
            scores = scores.view((batch_size * beam_size, 1))   # (batch_size * beam_size, 1)
            ids = ids.view((batch_size * beam_size, 1)) # (batch_size * beam_size, 1)

            new_hidden = None
            for i in range(0, batch_size):
                copy_target = hidden[0][:, i:i+1, :]
                copy_res = copy_target
                for j in range(0, beam_size - 1):
                    copy_res = torch.cat((copy_res, copy_target), dim=1)
                if new_hidden is None:
                    new_hidden = copy_res
                else:
                    new_hidden = torch.cat((new_hidden, copy_res), dim=1)
            assert new_hidden.shape[1] == batch_size * beam_size
            new_c = None
            for i in range(0, batch_size):
                copy_target = hidden[1][:, i:i+1, :]
                copy_res = copy_target
                for j in range(0, beam_size - 1):
                    copy_res = torch.cat((copy_res, copy_target), dim=1)
                if new_c is None:
                    new_c = copy_res
                else:
                    new_c = torch.cat((new_c, copy_res), dim=1)
            assert new_c.shape[1] == batch_size * beam_size
            hidden = (new_hidden, new_c)

            new_encoder_outputs = None
            for i in range(0, batch_size):
                copy_target = encoder_outputs[i:i+1, :, :]
                copy_res = copy_target
                for j in range(0, beam_size - 1):
                    copy_res = torch.cat((copy_res, copy_target), dim=0)
                if new_encoder_outputs is None:
                    new_encoder_outputs = copy_res
                else:
                    new_encoder_outputs = torch.cat((new_encoder_outputs, copy_res), dim=0)
            assert new_encoder_outputs.shape[0] == batch_size * beam_size
            encoder_outputs = new_encoder_outputs

            length = 1

            while True:
                eos_indices = _check_eos(ids[:, ids.shape[1] - 1], eos_id)
                if len(eos_indices) > 0:
                    for index in eos_indices:
                        batch_id = int(index / beam_size)
                        if len(res[batch_id]) < beam_size:
                            res[batch_id].append((scores[index][0], ids[index]))
                            if len(res[batch_id]) == beam_size:
                                ok += 1
                                is_done[batch_id] = True
                length += 1
                if ok == batch_size:
                    break
                if length == max_length:
                    break
                outputs = self.decoder(inputs=ids, hidden=hidden, encoder_outputs=encoder_outputs)  # (batch_size * beam_size, length, vocab_size)
                outputs = outputs[:, outputs.shape[1] - 1, :]   # (batch_size * beam_size, vocab_size)
                outputs = functional.log_softmax(outputs, dim=1)    # (batch_size * beam_size, vocab_size)

                for index in eos_indices:
                    outputs[index] = torch.zeros((self.vocab_size,))

                outputs = scores + outputs  # (batch_size * beam_size, vocab_size)
                outputs = outputs.view(batch_size, beam_size * self.vocab_size)
                scores, next_ids = torch.topk(outputs, beam_size, dim=1, largest=True)
                scores = scores.view(batch_size * beam_size, 1)
                ids = self._update_ids(ids, next_ids)

            scores = scores.view(batch_size, beam_size)
            if ok < batch_size:
                for batch_id in range(0, batch_size):
                    if is_done[batch_id]:
                        continue
                    need = beam_size - len(res[batch_id])
                    assert need > 0
                    for i in range(0, need):
                        res[batch_id].append((scores[batch_id][i], ids[batch_id * beam_size + i]))

            res_scores = None
            res_words = None
            assert len(res) == batch_size
            for batch_id in range(0, batch_size):
                assert len(res[batch_id]) == beam_size
                t_scores = None
                t_words = None
                for beam_id in range(0, beam_size):
                    item = res[batch_id][beam_id]
                    pad_length = length - item[1].shape[0]
                    new_item = None
                    if pad_length > 0:
                        new_item = (item[0], torch.cat((item[1], torch.full((pad_length,), pad_id, device=encoder_inputs.device)), dim=-1))
                    if new_item is not None:
                        item = new_item
                    if t_scores is None:
                        t_scores = torch.tensor(data=[item[0]], dtype=scores.dtype, device=scores.device)    # (1)
                    else:
                        t_scores = torch.cat((t_scores, torch.tensor(data=[item[0]], dtype=scores.dtype, device=scores.device)))
                    if t_words is None:
                        t_words = torch.unsqueeze(item[1], dim=0)   #(1, length)
                    else:
                        t_words = torch.cat((t_words, torch.unsqueeze(item[1], dim=0)), dim=0)
                assert t_scores.shape[0] == beam_size
                assert t_words.shape[0] == beam_size and t_words.shape[1] == length
                if res_scores is None:
                    res_scores = torch.unsqueeze(t_scores, dim=0)
                else:
                    res_scores = torch.cat((res_scores, torch.unsqueeze(t_scores, dim=0)), dim=0)
                if res_words is None:
                    res_words = torch.unsqueeze(t_words, dim=0)
                else:
                    res_words = torch.cat((res_words, torch.unsqueeze(t_words, dim=0)), dim=0)

            assert res_scores.shape[0] == batch_size and res_scores.shape[1] == beam_size
            assert res_words.shape[0] == batch_size and res_words.shape[1] == beam_size and res_words.shape[2] == length
            return res_scores, res_words

    def _update_ids(self, ids: torch.LongTensor, next_ids: torch.Tensor):
        """

        :param ids: (batch_size * beam_size, length)
        :param next_ids: (batch_size, beam_size)
        :return: res: (batch_size * beam_size, length + 1)
        """
        arr = []
        batch_size = next_ids.shape[0]
        beam_size = next_ids.shape[1]

        for i in range(0, batch_size):
            for j in range(0, beam_size):
                index_ids = int(i * (int(next_ids[i][j] / self.vocab_size)))
                index_vocab = int(next_ids[i][j] / beam_size)
                assert 0 <= index_ids < batch_size * beam_size
                assert 0 <= index_vocab < self.vocab_size
                arr.append((index_ids, index_vocab))

        res = None
        for val in arr:
            cur_ids = ids[val[0]]   # (length)
            cur_ids = torch.cat((cur_ids, torch.tensor(data=[val[1]], device=ids.device, dtype=torch.long)), dim=-1)  # (length + 1)
            if res is None:
                res = torch.unsqueeze(cur_ids, dim=0)   # (1, length + 1)
            else:
                res = torch.cat((res, torch.unsqueeze(cur_ids, dim=0)), dim=0)

        assert res.shape[0] == batch_size * beam_size
        assert res.shape[1] == ids.shape[1] + 1
        return res


def model_test():
    import numpy as np
    model = MyModel(embedding_dim=30, vocab_size=100).cuda()
    model.train(True)
    batch_size = 2
    encoder_length = 32
    decoder_length = 1
    encoder_inputs = torch.LongTensor(np.random.randint(0, 100, size=(batch_size, encoder_length))).cuda()
    decoder_inputs = torch.LongTensor(np.random.randint(0, 100, size=(batch_size, decoder_length))).cuda()
    outputs = model(encoder_inputs, decoder_inputs)
    print(outputs.shape)


def model_beam_search_test():
    model = MyModel(embedding_dim=30, vocab_size=103).cuda()
    model.train(False)
    batch_size = 2
    beam_size = 3
    encoder_length = 32
    encoder_inputs = torch.randint(size=(batch_size, encoder_length), high=103, device='cuda')
    scores, words = model(encoder_inputs, beam_size=beam_size, sos_id=100, eos_id=101, pad_id=102, max_length=64)
    print(scores)
    print(words)


def topk_test():
    x = torch.Tensor([[1, 4, 3, 2], [4, 2, 5, 1]])
    print(torch.topk(x, k=3))


def add_op_test():
    a = torch.Tensor([[0, 1, 2], [0, 1, 2]])
    b = torch.Tensor([[1], [2]])
    print(a[:,2:3])
    c = 3
    d = 2
    print(int(c / d))
    a[0] = torch.zeros((3,))
    print(a.shape[0])


if __name__ == '__main__':
    model_beam_search_test()
