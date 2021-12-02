import torch
from torch import nn
from time import perf_counter


def main():
    rnn = nn.RNN(input_size=1, hidden_size=2, batch_first=True, num_layers=2, bidirectional=True).cuda()
    # for name, param in rnn.named_parameters():
    #     print(name, param.data)
    h0 = torch.randn(4, 1, 2).cuda()
    # c0 = torch.randn(2, 100000, 2).cuda()
    input = torch.randn(1, 10, 1).cuda()
    start = perf_counter()
    # print(input)
    output, hn = rnn(input, h0)
    end = perf_counter()
    print(end - start)
    print(output[0])
    print(str(hn.size()))
    print(hn[0][0], hn[1][0], hn[2][0], hn[3][0])
    # for name, param in rnn.named_parameters():
    #     print(name, param.data)


if __name__ == '__main__':
    # a = torch.Tensor([[1, 2], [3, 4]])
    # print(a[0])
    temp = torch.Tensor()
    t1 = torch.Tensor([[1, 2, 3], [2, 3, 4]])
    t2 = torch.Tensor([[2, 3, 4], [3, 4, 5]])
    print(t1 + t2)
    temp = torch.cat((temp, t1), 0)
    temp = torch.cat((temp, t2), 0)
    print(temp)
    print(torch.unsqueeze(temp, 0))
    # main()
