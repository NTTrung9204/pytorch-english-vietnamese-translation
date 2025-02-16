import torch
from utils import build_vocab, train, evaluate
from Dataset import TranslationDatasetFull
from build_model import Transformer
from torchsummary import summary
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

if __name__ == "__main__":
    EN_FILE_PATH = "dataset/en_sents"
    VI_FILE_PATH = "dataset/vi_sents"
    BATCH = 32
    NUM_EPOCHS = 10
    DIM_MODEL = 512
    LEARNING_RATE = 0.0001

    en_vocab = build_vocab(EN_FILE_PATH, max_len=5000)
    vi_vocab = build_vocab(VI_FILE_PATH, max_len=5000)
    
    N_HEADS = 8
    N_LAYERS = 4
    D_FF = 512
    EN_VOCAB_SIZE = len(en_vocab) 
    VI_VOCAB_SIZE = len(vi_vocab) 
    DROPOUT = 0.1
    PAD_TOKEN_ID = vi_vocab['<pad>']

    print(f"English vocab size: {EN_VOCAB_SIZE}, Vietnamese vocab size: {VI_VOCAB_SIZE}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = TranslationDatasetFull(EN_FILE_PATH, VI_FILE_PATH, en_vocab, vi_vocab, max_len=50)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    model = Transformer(EN_VOCAB_SIZE, VI_VOCAB_SIZE, DIM_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.98))

    train_losses, valid_accuracies, valid_losses = train(model, train_dataloader, val_dataloader, criterion, optimizer, device, PAD_TOKEN_ID, NUM_EPOCHS)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color='blue')
    plt.plot(valid_losses, label="Validation Loss", color='red', linestyle='--')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_accuracies, label="Validation Accuracy", color="orange")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.show()

    torch.save(model.state_dict(), "trained_model.pth")
    print("Model saved successfully.")