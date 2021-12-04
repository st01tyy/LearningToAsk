import numpy as np
import struct


def main(embedding_dim: int, use_pretrained_embeddings: bool = True):
    FILE_PATH = 'pretrained/glove.6B.300d.txt'
    print('reading pretrained glove embeddings')
    if use_pretrained_embeddings:
        print('will save pretrained embeddings')
    else:
        print('will only save tokens')
    # map_token_id = {}
    tokens = []
    embeddings = []
    index = 0
    with open(FILE_PATH) as file_object:
        for line in file_object:
            line = line.rstrip()
            temp = line.split(' ')
            length = len(temp)
            if length != embedding_dim + 1:
                print('data structure mismatch, length of the line: ' + str(length) + ', embedding dim: '
                      + str(embedding_dim))
                continue
            # map_token_id[temp[0]] = index
            tokens.append(temp[0])
            if use_pretrained_embeddings:
                temp_embedding = []
                for i in range(1, length):
                    temp_embedding.append(float(temp[i]))
                embeddings.append(temp_embedding)
            index += 1
    print('read complete')
    temp_tokens = tokens[0:45000]
    temp_tokens.append(tokens[len(tokens) - 1])
    temp_embeddings = embeddings[0:45000]
    temp_embeddings.append(embeddings[len(embeddings) - 1])
    tokens = temp_tokens
    embeddings = temp_embeddings
    SOS_TOKEN = '<sos>'
    EOS_TOKEN = '<eos>'
    PAD_TOKEN = '<pad>'
    tokens.append(SOS_TOKEN)
    tokens.append(EOS_TOKEN)
    tokens.append(PAD_TOKEN)
    # map_token_id[SOS_TOKEN] = index
    # map_token_id[EOS_TOKEN] = index + 1
    if use_pretrained_embeddings:
        embeddings.append(np.random.rand(embedding_dim).tolist())
        embeddings.append(np.random.rand(embedding_dim).tolist())
        embeddings.append(np.random.rand(embedding_dim).tolist())
    # index += 3
    print('vocab size: ' + str(len(tokens)))

    TOKEN_PATH = 'vocab/tokens.txt'
    print('saving tokens')
    with open(TOKEN_PATH, mode='w') as file_object:
        for i in range(0, len(tokens)):
            file_object.write(tokens[i] + '\n')
    print('tokens saved')

    if not use_pretrained_embeddings:
        return
    EMBEDDING_PATH = 'vocab/embeddings.pth'
    print('saving pretrained embeddings')
    with open(EMBEDDING_PATH, mode='wb') as file_object:
        for i in range(0, len(embeddings)):
            file_object.write(struct.pack(str(embedding_dim) + 'd', *(embeddings[i])))
    print('embeddings saved')


def test():
    EMBEDDING_PATH = 'vocab/embeddings.pth'
    with open(EMBEDDING_PATH, mode='rb') as file_object:
        data = file_object.read(32)
        print(list(struct.unpack('4d', data)))


if __name__ == '__main__':
    main(embedding_dim=300)
