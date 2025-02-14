import torch
from collections import Counter
import string
import sys
from sklearn.metrics import accuracy_score

def build_vocab(file_path, max_len=10000):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = f.readlines()

    all_words = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip().lower()
        sentence = sentence.translate(str.maketrans("", "", string.punctuation))
        all_words.extend(sentence.split())

        sys.stdout.write(f"\rProcessing {file_path}, {index + 1:6d} | {len(sentences)}")

    word_counts = Counter(all_words)
    
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.most_common(max_len), 1)}
    vocab['<unk>'] = 0
    vocab['<pad>'] = len(vocab)
    vocab['<seqstart>'] = len(vocab)
    vocab['<seqend>'] = len(vocab)
    print()
    return vocab

def train(model, train_loader, valid_loader, criterion, optimizer, device, pad_token_id, num_epochs=10):
    """
    model: Transformer model nhận vào (encoder_input, decoder_input)
    train_loader, valid_loader: DataLoader của dataset huấn luyện và validation (mỗi mẫu là (encoder_input, decoder_input, target))
    criterion: hàm loss
    optimizer: optimizer của model
    device: thiết bị chạy (cpu hoặc cuda)
    pad_token_id: id của token <pad> (để xác định độ dài thực của decoder input)
    num_epochs: số epoch huấn luyện
    """
    train_losses = []
    valid_accuracies = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Mỗi batch có 3 phần: encoder_input, decoder_input, target
            encoder_input, decoder_input, target = batch
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            target = target.to(device).long()
            
            optimizer.zero_grad()

            # Forward: model nhận (src, tgt) và trả về logits có shape (batch, max_len, vocab_size)
            outputs = model(encoder_input, decoder_input)
            # Tính độ dài thực của decoder_input (số token khác pad)
            dec_lengths = (decoder_input != pad_token_id).sum(dim=1)
            # Lấy logits tại vị trí cuối (tương ứng với token cần dự đoán)
            batch_indices = torch.arange(outputs.size(0), device=device)
            # Vì dec_lengths là số token của prefix, nên vị trí cần dự đoán là dec_lengths - 1
            last_token_logits = outputs[batch_indices, dec_lengths - 1, :]  # shape: (batch, vocab_size)
            
            loss = criterion(last_token_logits, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            sys.stdout.write(f"\rEpoch [{epoch + 1:4d}/{num_epochs}] Batch [{batch_idx + 1}/{total_batches}], Loss: {loss.item():.4f}")
            sys.stdout.flush()

        avg_train_loss = running_loss / total_batches
        train_losses.append(avg_train_loss)

        valid_accuracy, valid_loss = evaluate(model, valid_loader, criterion, device, pad_token_id)
        valid_accuracies.append(valid_accuracy)
        valid_losses.append(valid_loss)

        print(f"\nEpoch [{epoch + 1:4d}/{num_epochs}], Avg Loss: {avg_train_loss:.4f}, Validation Accuracy: {valid_accuracy:.4f}, Validation Loss: {valid_loss:.4f}")

    return train_losses, valid_accuracies, valid_losses

def evaluate(model, valid_loader, criterion, device, pad_token_id):
    """
    Tương tự như train, evaluate chạy trên validation set để tính loss và accuracy.
    Mỗi batch gồm (encoder_input, decoder_input, target). Ta lấy logits tại vị trí cuối cùng của decoder_input.
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0

    with torch.no_grad():
        for batch in valid_loader:
            encoder_input, decoder_input, target = batch
            encoder_input = encoder_input.to(device)
            decoder_input = decoder_input.to(device)
            target = target.to(device).long()

            outputs = model(encoder_input, decoder_input)
            dec_lengths = (decoder_input != pad_token_id).sum(dim=1)
            batch_indices = torch.arange(outputs.size(0), device=device)
            last_token_logits = outputs[batch_indices, dec_lengths - 1, :]  # shape: (batch, vocab_size)

            loss = criterion(last_token_logits, target)
            running_loss += loss.item()

            _, predicted = torch.max(last_token_logits, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(target.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    avg_valid_loss = running_loss / len(valid_loader)
    return accuracy, avg_valid_loss
