import torch
from torch import nn
from time import perf_counter
import vocab_loader
from rnn_rnn import my_model
from typing import Dict


def tokenize(src: str, map_token_id: Dict):
    tokens = [map_token_id['<sos>']]
    src = src.split(' ')
    for i in range(0, len(src)):
        if map_token_id.__contains__(src[i]):
            tokens.append(map_token_id[src[i]])
        else:
            tokens.append(map_token_id['<unk>'])
    tokens.append(map_token_id['<eos>'])
    return tokens


def main():
    tokens, embeddings = vocab_loader.load_vocab()
    map_token_id = {}
    for i in range(0, len(tokens)):
        map_token_id[tokens[i]] = int(i)
    model = my_model.MyModel(300, len(tokens)).cuda()
    information = 'my name is ruben, i am 22 years old.'
    question = 'how old is ruben?'
    tokenized_information = tokenize(information, map_token_id)
    tokenized_question = tokenize(question, map_token_id)
    print(tokenized_information)
    print(tokenized_question)
    inputs = torch.LongTensor(tokenized_information)
    inputs = torch.unsqueeze(inputs, dim=0)
    targets = torch.LongTensor(tokenized_question)
    targets = torch.unsqueeze(targets, dim=0)
    outputs = model(inputs=inputs.cuda(), targets=targets.cuda())
    print(outputs)
    outputs = torch.squeeze(outputs)
    outputs = outputs.cpu()
    arr = outputs.numpy().tolist()
    for i in range(0, len(arr)):
        print(tokens[int(arr[i])] + ' ')


if __name__ == '__main__':
    main()
    # a = torch.Tensor([[1, 2, 3, 4],
    #                  [5, 6, 7, 8],
    #                   [1, 3, 5, 7]])
    # print(torch.argmax(input=a, dim=-1))
    # a = torch.Tensor([
    #     [[1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1]]
    # ])
    # b = torch.transpose(input=a, dim0=1, dim1=2)
    # a = a.bmm(b)
    # print(a)
    # temp = torch.Tensor()
    # for i in range(a.size()[0]):
    #     t = a[i].t()
    #     temp = torch.cat((temp, torch.unsqueeze(t, dim=0)), dim=0)
    # print(temp)
    # print(a.mm(temp))
    # print(temp)
    # a = torch.Tensor([[1, 2], [3, 4]])
    # print(a[0])
    # temp = torch.Tensor()
    # t1 = torch.Tensor([[1, 2, 3], [2, 3, 4]])
    # t2 = torch.Tensor([[2, 3, 4], [3, 4, 5]])
    # print(t1 + t2)
    # temp = torch.cat((temp, t1), 0)
    # temp = torch.cat((temp, t2), 0)
    # print(temp)
    # print(torch.unsqueeze(temp, 0))
    # main()
