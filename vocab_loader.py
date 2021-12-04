import struct


def load_vocab(load_pretrained_embeddings: bool = True, embedding_dim: int = 300):
    VOCAB_PATH = '../vocab/tokens.txt'
    EMBEDDINGS_PATH = '../vocab/embeddings.pth'
    tokens = []
    with open(VOCAB_PATH, mode='r') as fp:
        for line in fp:
            line = line.rstrip()
            if len(line) > 0:
                tokens.append(line)
    print(len(tokens))
    if load_pretrained_embeddings:
        print('loading pretrained embeddings')
        embeddings = []
        with open(EMBEDDINGS_PATH, mode='rb') as fp:
            data = fp.read(embedding_dim * 8)
            while data:
                embeddings.append(list(struct.unpack(str(embedding_dim) + 'd', data)))
                data = fp.read(embedding_dim * 8)
        print(len(embeddings))
        return tokens, embeddings
    return tokens


if __name__ == '__main__':
    load_vocab(load_pretrained_embeddings=False)
