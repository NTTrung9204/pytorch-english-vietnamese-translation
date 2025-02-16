import string
import torch
from torch.utils.data import Dataset

class TranslationDatasetFull(Dataset):
    def __init__(self, en_file, vi_file, en_vocab, vi_vocab, max_len=50):
        self.en_sentences = self._load_sentences(en_file)
        self.vi_sentences = self._load_sentences(vi_file)
        self.en_vocab = en_vocab
        self.vi_vocab = vi_vocab
        self.max_len = max_len

    def _load_sentences(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            sentences = f.readlines()[:10000]
        return sentences

    def _encode_sentence(self, sentence, vocab, max_len):
        sentence = sentence.strip().lower()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        tokens = sentence.split()
        token_ids = [vocab.get(token, vocab['<unk>']) for token in tokens]
        token_ids = token_ids[:max_len] + [vocab['<pad>']] * (max_len - len(token_ids))
        return token_ids

    def _encode_decoder_sentence(self, sentence, vocab, max_len):
        """
        Tạo chuỗi đầy đủ cho decoder: [<seqstart>] + tokens + [<seqend>]
        Sau đó padding nếu cần để đạt độ dài max_len.
        """
        sentence = sentence.strip().lower()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        tokens = sentence.split()
        full_tokens = [vocab['<seqstart>']] + [vocab.get(token, vocab['<unk>']) for token in tokens] + [vocab['<seqend>']]
        if len(full_tokens) < max_len:
            full_tokens += [vocab['<pad>']] * (max_len - len(full_tokens))
        else:
            full_tokens = full_tokens[:max_len]
        return full_tokens

    def __len__(self):
        return len(self.en_sentences)

    def __getitem__(self, idx):
        en_sentence = self.en_sentences[idx]
        vi_sentence = self.vi_sentences[idx]
        src = self._encode_sentence(en_sentence, self.en_vocab, self.max_len)
        tgt = self._encode_decoder_sentence(vi_sentence, self.vi_vocab, self.max_len)
        return torch.tensor(src, dtype=torch.long), torch.tensor(tgt, dtype=torch.long)
