import torch
from torch import nn
from time import perf_counter


def main():
    rnn = nn.RNN(input_size=1, hidden_size=2, batch_first=True, num_layers=1).cuda()
    # for name, param in rnn.named_parameters():
    #     print(name, param.data)
    h0 = torch.randn(1, 100000, 2).cuda()
    c0 = torch.randn(1, 100000, 2).cuda()
    input = torch.randn(100000, 10, 1).cuda()
    start = perf_counter()
    # print(input)
    output, hn = rnn(input, h0)
    end = perf_counter()
    print(end - start)
    # print(output)
    # for name, param in rnn.named_parameters():
    #     print(name, param.data)


if __name__ == '__main__':
    main()
