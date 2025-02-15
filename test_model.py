import torch
import string
from utils import build_vocab  # Giả sử build_vocab được định nghĩa trong utils
from Dataset import TranslationStepDataset
from build_model import Transformer

# Các tham số cấu hình
EN_FILE_PATH = "dataset/en_sents"
VI_FILE_PATH = "dataset/vi_sents"
BATCH = 32
NUM_EPOCHS = 10
DIM_MODEL = 512
LEARNING_RATE = 0.01
N_HEADS = 4
N_LAYERS = 4
D_FF = 512
DROPOUT = 0.1

# Xây dựng vocab cho tiếng Anh và tiếng Việt
en_vocab = build_vocab(EN_FILE_PATH, max_len=5000)
vi_vocab = build_vocab(VI_FILE_PATH, max_len=5000)
EN_VOCAB_SIZE = len(en_vocab)
VI_VOCAB_SIZE = len(vi_vocab)
PAD_TOKEN_ID = vi_vocab['<pad>']

print(f"English vocab size: {EN_VOCAB_SIZE}, Vietnamese vocab size: {VI_VOCAB_SIZE}")

# Thiết bị chạy (CPU hoặc GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Khởi tạo dataset (để sử dụng hàm _encode_sentence)
dataset = TranslationStepDataset(EN_FILE_PATH, VI_FILE_PATH, en_vocab, vi_vocab, max_len=50)

# Load mô hình đã train
model = Transformer(EN_VOCAB_SIZE, VI_VOCAB_SIZE, DIM_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT)
model.load_state_dict(torch.load("trained_model.pth", map_location=device))
model = model.to(device)


def translate_sentence(model, sentence, en_vocab, vi_vocab, device, max_len=50, dataset_obj=None):
    """
    Hàm dịch một câu tiếng Anh sang tiếng Việt sử dụng giải mã greedy.
    
    Args:
        model: Mô hình Transformer đã train.
        sentence: Câu tiếng Anh đầu vào (str).
        en_vocab: Từ điển tiếng Anh (token -> id).
        vi_vocab: Từ điển tiếng Việt (token -> id).
        device: torch.device.
        max_len: Độ dài cố định cho encoder input và decoder input.
        dataset_obj: (Tùy chọn) Instance của TranslationStepDataset để sử dụng hàm _encode_sentence.
        
    Returns:
        Dịch tiếng Việt (str).
    """
    model.eval()

    # Encode câu tiếng Anh. Nếu có dataset_obj, dùng hàm _encode_sentence của nó
    if dataset_obj is not None:
        src_ids = dataset_obj._encode_sentence(sentence, en_vocab, max_len)
    else:
        # Nếu không có, ta tự tiền xử lý
        sentence_proc = sentence.strip().lower()
        sentence_proc = sentence_proc.translate(str.maketrans("", "", string.punctuation))
        tokens = sentence_proc.split()
        src_ids = [en_vocab.get(token, en_vocab['<unk>']) for token in tokens]
        src_ids = src_ids[:max_len] + [en_vocab['<pad>']] * (max_len - len(src_ids))
        
    src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0).to(device)  # shape: (1, max_len)

    print(src_tensor)

    # Khởi tạo decoder input với token <seqstart>
    decoder_input = [vi_vocab['<seqstart>']]

    print(decoder_input)

    for i in range(max_len - 1):
        # Padding decoder input về độ dài cố định max_len
        dec_input = decoder_input + [vi_vocab['<pad>']] * (max_len - len(decoder_input))
        dec_tensor = torch.tensor(dec_input, dtype=torch.long).unsqueeze(0).to(device)  # shape: (1, max_len)
        
        # Mô hình nhận (encoder_input, decoder_input) và trả về logits có shape (1, max_len, VI_VOCAB_SIZE)
        outputs = model(src_tensor, dec_tensor)
        print(outputs.shape)
        # Lấy logits tại vị trí cuối của chuỗi decoder input hiện tại
        current_index = len(decoder_input) - 1
        logits = outputs[0, current_index, :]  # shape: (VI_VOCAB_SIZE,)
        predicted_id = torch.argmax(logits).item()
        decoder_input.append(predicted_id)

        print(decoder_input)

        # Nếu dự đoán được token <seqend>, dừng giải mã
        if predicted_id == vi_vocab['<seqend>']:
            break

    # Đảo từ điển vi_vocab để mapping id -> token
    inv_vi_vocab = {id: token for token, id in vi_vocab.items()}
    predicted_tokens = [inv_vi_vocab.get(tok, '<unk>') for tok in decoder_input]

    print(predicted_tokens)

    # Loại bỏ token <seqstart> và các token sau <seqend>
    if predicted_tokens[0] == '<seqstart>':
        predicted_tokens = predicted_tokens[1:]
    if '<seqend>' in predicted_tokens:
        predicted_tokens = predicted_tokens[:predicted_tokens.index('<seqend>')]

    return ' '.join(predicted_tokens)


# Test mô hình với một câu tiếng Anh
test_sentence = "I love you so much"
translation = translate_sentence(model, test_sentence, en_vocab, vi_vocab, device, max_len=50, dataset_obj=dataset)
print("Input sentence: ", test_sentence)
print("Translated sentence: ", translation)
