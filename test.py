from utils import build_vocab
from Dataset import TranslationStepDataset
from build_model import Transformer
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader


# dataset = TranslationStepDataset('dataset/en_sents', 'dataset/vi_sents', en_vocab, vi_vocab, max_len=50)

# print(dataset[0])
# print(dataset[1])


# Ví dụ cách sử dụng:
if __name__ == "__main__":
    en_file = "dataset/en_sents"
    vi_file = "dataset/vi_sents"

    en_vocab = build_vocab(en_file, max_len=10000)
    vi_vocab = build_vocab(vi_file, max_len=10000)

    # print(vi_vocab)

    dataset = TranslationStepDataset(en_file, vi_file, en_vocab, vi_vocab, max_len=50)

    print(len(dataset))

    inv_vi_vocab = {id: token for token, id in vi_vocab.items()}

    for batch in dataset:
        encoder_input, decoder_input, target = batch
        print(encoder_input)
        print(decoder_input)
        print(inv_vi_vocab[target.item()])
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # # Lấy 1 batch và in ra các tensor
    # for batch in dataloader:
    #     src, dec_input, target = batch
    #     print("Encoder input shape:", src.shape)       # (batch_size, max_len)
    #     print("Decoder input shape:", dec_input.shape)   # (batch_size, max_len)
    #     print("Target shape:", target.shape)             # (batch_size,)
    #     # In vài mẫu kiểm tra:
    #     for i in range(min(3, src.size(0))):
    #         print("Encoder:", src[i].tolist())
    #         print("Decoder input:", dec_input[i].tolist())
    #         print("Target:", target[i].item())
    #         print("-----")
    #     break

# # Tham số mô hình
# d_model = 512
# n_heads = 8
# n_layers = 6
# d_ff = 2048
# en_vocab_size = len(en_vocab) 
# vi_vocab_size = len(vi_vocab) 
# dropout = 0.1

# model = Transformer(en_vocab_size, vi_vocab_size, d_model, n_heads, n_layers, d_ff, dropout)

# summary(model, [(50,), (50,)], device="cpu")

# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#          Embedding-1              [-1, 50, 512]       5,121,024
#            Dropout-2              [-1, 50, 512]               0
#             Linear-3              [-1, 50, 512]         262,656
#             Linear-4              [-1, 50, 512]         262,656
#             Linear-5              [-1, 50, 512]         262,656
#             Linear-6              [-1, 50, 512]         262,656
# MultiHeadAttention-7              [-1, 50, 512]               0
#          LayerNorm-8              [-1, 50, 512]           1,024
#             Linear-9             [-1, 50, 2048]       1,050,624
#           Dropout-10             [-1, 50, 2048]               0
#            Linear-11              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-12              [-1, 50, 512]               0
#         LayerNorm-13              [-1, 50, 512]           1,024
#      EncoderLayer-14              [-1, 50, 512]               0
#            Linear-15              [-1, 50, 512]         262,656
#            Linear-16              [-1, 50, 512]         262,656
#            Linear-17              [-1, 50, 512]         262,656
#            Linear-18              [-1, 50, 512]         262,656
# MultiHeadAttention-19              [-1, 50, 512]               0
#         LayerNorm-20              [-1, 50, 512]           1,024
#            Linear-21             [-1, 50, 2048]       1,050,624
#           Dropout-22             [-1, 50, 2048]               0
#            Linear-23              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-24              [-1, 50, 512]               0
#         LayerNorm-25              [-1, 50, 512]           1,024
#      EncoderLayer-26              [-1, 50, 512]               0
#            Linear-27              [-1, 50, 512]         262,656
#            Linear-28              [-1, 50, 512]         262,656
#            Linear-29              [-1, 50, 512]         262,656
#            Linear-30              [-1, 50, 512]         262,656
# MultiHeadAttention-31              [-1, 50, 512]               0
#         LayerNorm-32              [-1, 50, 512]           1,024
#            Linear-33             [-1, 50, 2048]       1,050,624
#           Dropout-34             [-1, 50, 2048]               0
#            Linear-35              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-36              [-1, 50, 512]               0
#         LayerNorm-37              [-1, 50, 512]           1,024
#      EncoderLayer-38              [-1, 50, 512]               0
#            Linear-39              [-1, 50, 512]         262,656
#            Linear-40              [-1, 50, 512]         262,656
#            Linear-41              [-1, 50, 512]         262,656
#            Linear-42              [-1, 50, 512]         262,656
# MultiHeadAttention-43              [-1, 50, 512]               0
#         LayerNorm-44              [-1, 50, 512]           1,024
#            Linear-45             [-1, 50, 2048]       1,050,624
#           Dropout-46             [-1, 50, 2048]               0
#            Linear-47              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-48              [-1, 50, 512]               0
#         LayerNorm-49              [-1, 50, 512]           1,024
#      EncoderLayer-50              [-1, 50, 512]               0
#            Linear-51              [-1, 50, 512]         262,656
#            Linear-52              [-1, 50, 512]         262,656
#            Linear-53              [-1, 50, 512]         262,656
#            Linear-54              [-1, 50, 512]         262,656
# MultiHeadAttention-55              [-1, 50, 512]               0
#         LayerNorm-56              [-1, 50, 512]           1,024
#            Linear-57             [-1, 50, 2048]       1,050,624
#           Dropout-58             [-1, 50, 2048]               0
#            Linear-59              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-60              [-1, 50, 512]               0
#         LayerNorm-61              [-1, 50, 512]           1,024
#      EncoderLayer-62              [-1, 50, 512]               0
#            Linear-63              [-1, 50, 512]         262,656
#            Linear-64              [-1, 50, 512]         262,656
#            Linear-65              [-1, 50, 512]         262,656
#            Linear-66              [-1, 50, 512]         262,656
# MultiHeadAttention-67              [-1, 50, 512]               0
#         LayerNorm-68              [-1, 50, 512]           1,024
#            Linear-69             [-1, 50, 2048]       1,050,624
#           Dropout-70             [-1, 50, 2048]               0
#            Linear-71              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-72              [-1, 50, 512]               0
#         LayerNorm-73              [-1, 50, 512]           1,024
#      EncoderLayer-74              [-1, 50, 512]               0
#           Encoder-75              [-1, 50, 512]               0
#         Embedding-76              [-1, 50, 512]       3,649,024
#           Dropout-77              [-1, 50, 512]               0
#            Linear-78              [-1, 50, 512]         262,656
#            Linear-79              [-1, 50, 512]         262,656
#            Linear-80              [-1, 50, 512]         262,656
#            Linear-81              [-1, 50, 512]         262,656
# MultiHeadAttention-82              [-1, 50, 512]               0
#         LayerNorm-83              [-1, 50, 512]           1,024
#            Linear-84              [-1, 50, 512]         262,656
#            Linear-85              [-1, 50, 512]         262,656
#            Linear-86              [-1, 50, 512]         262,656
#            Linear-87              [-1, 50, 512]         262,656
# MultiHeadAttention-88              [-1, 50, 512]               0
#         LayerNorm-89              [-1, 50, 512]           1,024
#            Linear-90             [-1, 50, 2048]       1,050,624
#           Dropout-91             [-1, 50, 2048]               0
#            Linear-92              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-93              [-1, 50, 512]               0
#         LayerNorm-94              [-1, 50, 512]           1,024
#      DecoderLayer-95              [-1, 50, 512]               0
#            Linear-96              [-1, 50, 512]         262,656
#            Linear-97              [-1, 50, 512]         262,656
#            Linear-98              [-1, 50, 512]         262,656
#            Linear-99              [-1, 50, 512]         262,656
# MultiHeadAttention-100              [-1, 50, 512]               0
#        LayerNorm-101              [-1, 50, 512]           1,024
#           Linear-102              [-1, 50, 512]         262,656
#           Linear-103              [-1, 50, 512]         262,656
#           Linear-104              [-1, 50, 512]         262,656
#           Linear-105              [-1, 50, 512]         262,656
# MultiHeadAttention-106              [-1, 50, 512]               0
#        LayerNorm-107              [-1, 50, 512]           1,024
#           Linear-108             [-1, 50, 2048]       1,050,624
#          Dropout-109             [-1, 50, 2048]               0
#           Linear-110              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-111              [-1, 50, 512]               0
#        LayerNorm-112              [-1, 50, 512]           1,024
#     DecoderLayer-113              [-1, 50, 512]               0
#           Linear-114              [-1, 50, 512]         262,656
#           Linear-115              [-1, 50, 512]         262,656
#           Linear-116              [-1, 50, 512]         262,656
#           Linear-117              [-1, 50, 512]         262,656
# MultiHeadAttention-118              [-1, 50, 512]               0
#        LayerNorm-119              [-1, 50, 512]           1,024
#           Linear-120              [-1, 50, 512]         262,656
#           Linear-121              [-1, 50, 512]         262,656
#           Linear-122              [-1, 50, 512]         262,656
#           Linear-123              [-1, 50, 512]         262,656
# MultiHeadAttention-124              [-1, 50, 512]               0
#        LayerNorm-125              [-1, 50, 512]           1,024
#           Linear-126             [-1, 50, 2048]       1,050,624
#          Dropout-127             [-1, 50, 2048]               0
#           Linear-128              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-129              [-1, 50, 512]               0
#        LayerNorm-130              [-1, 50, 512]           1,024
#     DecoderLayer-131              [-1, 50, 512]               0
#           Linear-132              [-1, 50, 512]         262,656
#           Linear-133              [-1, 50, 512]         262,656
#           Linear-134              [-1, 50, 512]         262,656
#           Linear-135              [-1, 50, 512]         262,656
# MultiHeadAttention-136              [-1, 50, 512]               0
#        LayerNorm-137              [-1, 50, 512]           1,024
#           Linear-138              [-1, 50, 512]         262,656
#           Linear-139              [-1, 50, 512]         262,656
#           Linear-140              [-1, 50, 512]         262,656
#           Linear-141              [-1, 50, 512]         262,656
# MultiHeadAttention-142              [-1, 50, 512]               0
#        LayerNorm-143              [-1, 50, 512]           1,024
#           Linear-144             [-1, 50, 2048]       1,050,624
#          Dropout-145             [-1, 50, 2048]               0
#           Linear-146              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-147              [-1, 50, 512]               0
#        LayerNorm-148              [-1, 50, 512]           1,024
#     DecoderLayer-149              [-1, 50, 512]               0
#           Linear-150              [-1, 50, 512]         262,656
#           Linear-151              [-1, 50, 512]         262,656
#           Linear-152              [-1, 50, 512]         262,656
#           Linear-153              [-1, 50, 512]         262,656
# MultiHeadAttention-154              [-1, 50, 512]               0
#        LayerNorm-155              [-1, 50, 512]           1,024
#           Linear-156              [-1, 50, 512]         262,656
#           Linear-157              [-1, 50, 512]         262,656
#           Linear-158              [-1, 50, 512]         262,656
#           Linear-159              [-1, 50, 512]         262,656
# MultiHeadAttention-160              [-1, 50, 512]               0
#        LayerNorm-161              [-1, 50, 512]           1,024
#           Linear-162             [-1, 50, 2048]       1,050,624
#          Dropout-163             [-1, 50, 2048]               0
#           Linear-164              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-165              [-1, 50, 512]               0
#        LayerNorm-166              [-1, 50, 512]           1,024
#     DecoderLayer-167              [-1, 50, 512]               0
#           Linear-168              [-1, 50, 512]         262,656
#           Linear-169              [-1, 50, 512]         262,656
#           Linear-170              [-1, 50, 512]         262,656
#           Linear-171              [-1, 50, 512]         262,656
# MultiHeadAttention-172              [-1, 50, 512]               0
#        LayerNorm-173              [-1, 50, 512]           1,024
#           Linear-174              [-1, 50, 512]         262,656
#           Linear-175              [-1, 50, 512]         262,656
#           Linear-176              [-1, 50, 512]         262,656
#           Linear-177              [-1, 50, 512]         262,656
# MultiHeadAttention-178              [-1, 50, 512]               0
#        LayerNorm-179              [-1, 50, 512]           1,024
#           Linear-180             [-1, 50, 2048]       1,050,624
#          Dropout-181             [-1, 50, 2048]               0
#           Linear-182              [-1, 50, 512]       1,049,088
# PositionwiseFeedForward-183              [-1, 50, 512]               0
#        LayerNorm-184              [-1, 50, 512]           1,024
#     DecoderLayer-185              [-1, 50, 512]               0
#           Linear-186             [-1, 50, 7127]       3,656,151
#          Decoder-187             [-1, 50, 7127]               0
# ================================================================
# Total params: 56,564,695
# Trainable params: 56,564,695
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.01
# Forward/backward pass size (MB): 55.63
# Params size (MB): 215.78
# Estimated Total Size (MB): 271.42
# ----------------------------------------------------------------