import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import string

# --------------------------
# Cấu hình chung
# --------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from utils import build_vocab

EN_FILE_PATH = "dataset/en_sents"
VI_FILE_PATH = "dataset/vi_sents"

DIM_MODEL = 512
N_HEADS = 8
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.1

en_vocab = build_vocab(EN_FILE_PATH, max_len=10000)
vi_vocab = build_vocab(VI_FILE_PATH, max_len=10000)
EN_VOCAB_SIZE = len(en_vocab)
VI_VOCAB_SIZE = len(vi_vocab)

BOS_IDX = vi_vocab['<seqstart>']  # token bắt đầu
EOS_IDX = vi_vocab['<seqend>']     # token kết thúc

print(f"English vocab size: {EN_VOCAB_SIZE}, Vietnamese vocab size: {VI_VOCAB_SIZE}")

# --------------------------
# Khởi tạo mô hình Transformer
# --------------------------
from build_model import Transformer

model = Transformer(EN_VOCAB_SIZE, VI_VOCAB_SIZE, DIM_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT)
model.load_state_dict(torch.load("trained_model_kaggle.pth", map_location=DEVICE))
model = model.to(DEVICE)

# Nếu lớp Transformer chưa có thuộc tính generator, ta có thể gán:
model.generator = model.decoder.output_linear

# --------------------------
# Hàm tạo subsequent mask
# --------------------------
def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

# --------------------------
# Hàm giải mã greedy
# --------------------------
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    """
    Giải mã greedy cho mô hình Transformer.
    
    Args:
        model: Transformer đã train.
        src: Tensor câu nguồn với shape [batch, src_seq_len] (ở đây batch = 1).
        src_mask: Mask cho encoder, shape [src_seq_len, src_seq_len].
        max_len: Độ dài tối đa của câu đầu ra.
        start_symbol: BOS token id.
        
    Returns:
        ys: Tensor các token id của câu đầu ra, shape [batch, output_seq_len].
    """
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)
    
    # Lấy memory từ encoder: memory có shape [batch, src_seq_len, d_model]
    memory = model.encoder(src, src_mask)
    
    # Khởi tạo chuỗi decoder với token bắt đầu; shape: [batch, 1]
    ys = torch.ones(src.size(0), 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    
    for i in range(max_len - 1):
        # Tạo mask cho decoder với kích thước [current_seq_len, current_seq_len]
        tgt_mask = generate_square_subsequent_mask(ys.size(1)).to(DEVICE)
        
        # Gọi decoder: model.decoder nhận vào (ys, memory, src_mask, tgt_mask, None, None)
        # Giả sử model.decoder trả về logits có shape [batch, tgt_seq_len, vocab_size]
        out = model.decoder(ys, memory, src_mask, tgt_mask, None, None)
        
        # Lấy logits của token cuối cùng: shape: [batch, vocab_size]
        out_last = out[:, -1, :]
        
        # Vì output của decoder đã có kích thước vocab_size, không cần gọi thêm model.generator
        prob = out_last  # prob có shape [batch, VI_VOCAB_SIZE]
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        
        # Ghép token mới vào chuỗi decoder: kết quả có shape [batch, current_seq_len+1]
        ys = torch.cat([ys, torch.tensor([[next_word]], device=DEVICE)], dim=1)
        
        # Nếu dự đoán được token kết thúc, dừng giải mã
        if next_word == EOS_IDX:
            break
            
    return ys


# --------------------------
# Hàm chuyển đổi câu nguồn thành tensor
# --------------------------
def simple_text_transform(sentence: str):
    sentence = sentence.strip().lower()
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    tokens = sentence.split()
    token_ids = [en_vocab.get(token, en_vocab['<unk>']) for token in tokens]
    return torch.tensor(token_ids, dtype=torch.long).unsqueeze(0)  # shape: [1, seq_len]

# --------------------------
# Lớp chuyển đổi từ id sang token
# --------------------------
class SimpleVocab:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {id: token for token, id in vocab.items()}
    def lookup_tokens(self, ids):
        return [self.inv_vocab.get(i, "<unk>") for i in ids]

vocab_transform = SimpleVocab(vi_vocab)

# --------------------------
# Hàm dịch câu (translate)
# --------------------------
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src_tensor = simple_text_transform(src_sentence).to(DEVICE)  # shape: [1, seq_len]
    seq_len = src_tensor.size(1)
    src_mask = torch.zeros(seq_len, seq_len, device=DEVICE).type(torch.bool)
    
    ys = greedy_decode(model, src_tensor, src_mask, max_len=seq_len + 5, start_symbol=BOS_IDX)
    tgt_tokens = ys.squeeze(0).cpu().numpy().tolist()
    
    tokens = vocab_transform.lookup_tokens(tgt_tokens)
    translation = " ".join(tokens).replace("<seqstart>", "").replace("<seqend>", "").strip()
    return translation

# --------------------------
# Test model
# --------------------------
if __name__ == "__main__":
    test_sentence = "What clothes do you think I should put on to go to my date tomorrow?"
    translation = translate(model, test_sentence)
    print("Input sentence:", test_sentence)
    print("Translated sentence:", translation)
