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
    Mỗi sample trong DataLoader trả về: (encoder_input, decoder_target) với shape [batch, seq_len].
    Trong đó decoder_target là chuỗi đầy đủ: [<seqstart>, token1, token2, ..., <seqend>].
    
    Chúng ta sử dụng teacher forcing:
      - tgt_input = decoder_target[:, :-1]
      - tgt_out   = decoder_target[:, 1:]
    """
    train_losses = []
    valid_accuracies = []
    valid_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, batch in enumerate(train_loader):
            # Mỗi batch: encoder_input, decoder_target (shape: [batch, seq_len])
            encoder_input, decoder_target = batch
            encoder_input = encoder_input.to(device)
            decoder_target = decoder_target.to(device).long()
            
            # Tách teacher forcing:
            tgt_input = decoder_target[:, :-1]  # [batch, seq_len-1]
            tgt_out   = decoder_target[:, 1:]    # [batch, seq_len-1]
            
            # Tạo mask dựa trên input có shape [batch, seq_len]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(encoder_input, tgt_input, device, pad_token_id)
            
            optimizer.zero_grad()

            # Forward: model nhận (encoder_input, tgt_input) và các mask, trả về logits với shape [batch, tgt_seq_len, vocab_size]
            outputs = model(encoder_input, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
            
            # Tính loss: reshape outputs và tgt_out thành vector 1 chiều
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tgt_out.reshape(-1))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            sys.stdout.write(f"\rEpoch [{epoch + 1:2d}/{num_epochs}] Batch [{batch_idx + 1}/{total_batches}], Loss: {loss.item():.4f}")
            sys.stdout.flush()

        avg_train_loss = running_loss / total_batches
        train_losses.append(avg_train_loss)

        valid_accuracy, valid_loss = evaluate(model, valid_loader, criterion, device, pad_token_id)
        valid_accuracies.append(valid_accuracy)
        valid_losses.append(valid_loss)

        print(f"\nEpoch [{epoch + 1:2d}/{num_epochs}], Avg Loss: {avg_train_loss:.4f}, Val Loss: {valid_loss:.4f}, Val Acc: {valid_accuracy:.4f}")

    return train_losses, valid_accuracies, valid_losses

def evaluate(model, valid_loader, criterion, device, pad_token_id):
    """
    evaluate chạy trên validation set để tính loss và accuracy.
    Mỗi batch trả về: encoder_input, decoder_target với shape [batch, seq_len].
    Chúng ta tách teacher forcing (tgt_input và tgt_out) và tạo mask.
    """
    model.eval()
    all_preds = []
    all_labels = []
    running_loss = 0.0
    
    with torch.no_grad():
        for batch in valid_loader:
            # Giả sử mỗi batch: encoder_input, decoder_target có shape [batch, seq_len]
            encoder_input, decoder_target = batch
            encoder_input = encoder_input.to(device)
            decoder_target = decoder_target.to(device)
            
            # Tách teacher forcing:
            # tgt_input: toàn bộ chuỗi ngoại trừ token cuối
            # tgt_out: toàn bộ chuỗi ngoại trừ token đầu (sẽ so sánh dự đoán cho mỗi bước)
            tgt_input = decoder_target[:, :-1]  # shape: [batch, seq_len-1]
            tgt_out   = decoder_target[:, 1:]    # shape: [batch, seq_len-1]
            
            # Tạo mask dựa trên dữ liệu có shape [batch, seq_len]
            # Lưu ý: create_mask phải được định nghĩa để nhận đầu vào dạng [batch, seq_len]
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(encoder_input, tgt_input, device, pad_token_id)
            
            # Gọi model với dữ liệu có shape [batch, seq_len] (không transpose)
            outputs = model(encoder_input, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask)
            # outputs có shape: [batch, seq_len-1, vocab_size]
            
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), tgt_out.reshape(-1))
            running_loss += loss.item()
            
            # Dự đoán: lấy argmax theo chiều vocab, kết quả có shape [batch, seq_len-1]
            _, predicted = torch.max(outputs, dim=-1)
            all_preds.extend(predicted.reshape(-1).cpu().numpy())
            all_labels.extend(tgt_out.reshape(-1).cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    avg_valid_loss = running_loss / len(valid_loader)
    return accuracy, avg_valid_loss



def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, DEVICE, PAD_IDX):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask
