import string
from torch.utils.data import Dataset, DataLoader
import torch

class TranslationDataset(Dataset):
    def __init__(self, en_file, vi_file, en_vocab, vi_vocab, max_len=50):
        self.en_sentences = self._load_sentences(en_file)
        self.vi_sentences = self._load_sentences(vi_file)
        self.en_vocab = en_vocab
        self.vi_vocab = vi_vocab
        self.max_len = max_len

    def _load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()
        return sentences

    def _encode_sentence(self, sentence, vocab, max_len):
        # Tiền xử lý, chuyển chữ thường và thêm <unk> cho từ không có trong vocab
        sentence = sentence.strip().lower()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))  # Loại bỏ dấu câu nếu cần
        tokens = sentence.split()
        token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
        
        # Cắt hoặc padding đến max_len
        token_ids = token_ids[:max_len] + [vocab['<pad>']] * (max_len - len(token_ids))
        return token_ids

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_sentence = self.en_sentences[idx]
        vi_sentence = self.vi_sentences[idx]
        en_tokens = self._encode_sentence(en_sentence, self.en_vocab, self.max_len)
        vi_tokens = self._encode_sentence(vi_sentence, self.vi_vocab, self.max_len)
        return torch.tensor(en_tokens), torch.tensor(vi_tokens)