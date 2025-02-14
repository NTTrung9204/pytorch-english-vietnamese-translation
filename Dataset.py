import torch
from torch.utils.data import Dataset, DataLoader
import string
import sys

class TranslationStepDataset(Dataset):
    def __init__(self, en_file, vi_file, en_vocab, vi_vocab, max_len=50):
        """
        en_file: file chứa các câu tiếng Anh (mỗi dòng một câu)
        vi_file: file chứa các câu tiếng Việt tương ứng
        en_vocab, vi_vocab: dict mapping token -> id, đã có các token đặc biệt như <unk>, <pad>, <seqstart>, <seqend>
        max_len: độ dài cố định cho encoder input và decoder input (decoder input sau padding luôn có độ dài max_len)
        """
        self.en_sentences = self._load_sentences(en_file)
        self.vi_sentences = self._load_sentences(vi_file)
        self.en_vocab = en_vocab
        self.vi_vocab = vi_vocab
        self.max_len = max_len
        self.samples = []
        self.build_samples()

    def _load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.readlines()

    def _encode_sentence(self, sentence, vocab, max_len):
        """
        Encode câu: chuyển về chữ thường, loại bỏ dấu câu, tách token và mapping sang id.
        Padding đến độ dài cố định max_len.
        """
        sentence = sentence.strip().lower()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        tokens = sentence.split()
        token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
        token_ids = token_ids[:max_len] + [vocab['<pad>']] * (max_len - len(token_ids))
        return token_ids

    def build_samples(self):
        """
        Với mỗi cặp câu (encoder: tiếng Anh, decoder: tiếng Việt),
        ta tạo ra các mẫu huấn luyện theo dạng step-by-step:
          - Encoder input luôn cố định: vd. [0, 1, 2, 3]
          - Decoder input: tiền tố của chuỗi token đã encode (đã thêm token <seqstart>) được padding về độ dài max_len
          - Target: là token kế tiếp (scalar)
        
        Ví dụ với câu tiếng Việt được encode thành: [<seqstart>, token1, token2, ..., <seqend>]
          * Mẫu 1: 
              - decoder_input: [<seqstart>, <pad>, <pad>, ..., <pad>]  (độ dài max_len)
              - target: token1
          * Mẫu 2:
              - decoder_input: [<seqstart>, token1, <pad>, ..., <pad>]
              - target: token2
          * ...
          * Mẫu cuối:
              - decoder_input: [<seqstart>, token1, token2, ..., token_k] (với k < max_len)
              - target: <seqend>
        """
        index = 0
        for en_sentence, vi_sentence in zip(self.en_sentences, self.vi_sentences):
            sys.stdout.write(f"\rBuilding data... {index + 1:6d} | {len(self.en_sentences)}")
            # Encode câu tiếng Anh cho encoder (độ dài cố định)
            src = self._encode_sentence(en_sentence, self.en_vocab, self.max_len)

            # Tiền xử lý câu tiếng Việt: lowercase, loại bỏ dấu câu, tách token
            vi_sentence = vi_sentence.strip().lower()
            vi_sentence = vi_sentence.translate(str.maketrans("", "", string.punctuation))
            tokens = vi_sentence.split()

            # Tạo chuỗi token đầy đủ cho decoder: [<seqstart>] + tokens + [<seqend>]
            full_tokens = [self.vi_vocab['<seqstart>']] + \
                          [self.vi_vocab.get(token, self.vi_vocab['<unk>']) for token in tokens] + \
                          [self.vi_vocab['<seqend>']]
            # Nếu chuỗi dài hơn max_len thì cắt bớt (sẽ bỏ đi các token sau)
            full_tokens = full_tokens[:self.max_len]

            # Với mỗi vị trí trong full_tokens (bắt đầu từ index 1), tạo mẫu:
            for i in range(1, len(full_tokens)):
                dec_input = full_tokens[:i]  # prefix hiện có
                target = full_tokens[i]        # token tiếp theo cần dự đoán

                # Padding decoder input về độ dài cố định max_len
                if len(dec_input) < self.max_len:
                    dec_input = dec_input + [self.vi_vocab['<pad>']] * (self.max_len - len(dec_input))

                self.samples.append((
                    torch.tensor(src, dtype=torch.long),         # encoder input (fixed length)
                    torch.tensor(dec_input, dtype=torch.long),     # decoder input (fixed length max_len)
                    torch.tensor(target, dtype=torch.long)         # target (scalar)
                ))

            index += 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


