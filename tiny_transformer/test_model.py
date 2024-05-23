import torch

from configuration import ModelConfig
from model import Transformer

from utils import subsequent_mask

config = ModelConfig(encoder_layer_nums=2, decoder_layer_nums=2)
print(config.__dict__)


def build_model():
    model = Transformer(config)
    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            torch.nn.init.xavier_uniform_(p)

    return model


def inference_test():
    test_model = build_model()
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    src_mask = torch.ones(1, 1, 10)
    memory = test_model.encode(src, src_mask)
    ys = torch.zeros(1, 1).type_as(src)
    for i in range(9):
        out = test_model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        prob = test_model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    print("Example Untrained Model Prediction:", ys)


def run_tests():
    for _ in range(10):
        inference_test()


run_tests()
