from utils import build_vocab
from Dataset import TranslationDataset

en_vocab = build_vocab('dataset/en_sents', max_len=10000)
vi_vocab = build_vocab('dataset/vi_sents', max_len=10000)

dataset = TranslationDataset('dataset/en_sents', 'dataset/vi_sents', en_vocab, vi_vocab, max_len=50)

print(dataset[0])
print(dataset[1])