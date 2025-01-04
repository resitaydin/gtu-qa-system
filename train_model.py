import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizerFast, BertForQuestionAnswering
from transformers import AdamW
from tqdm.auto import tqdm
import numpy as np

class QADataset(Dataset):
    def __init__(self, data, tokenizer, max_len=256):  
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        # Reserve tokens for [CLS] and [SEP]
        self.max_len_per_segment = (self.max_len - 3) // 2  # Account for [CLS] and two [SEP] tokens

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = str(self.data.iloc[idx]['question'])
        context = str(self.data.iloc[idx]['context'])
        answer = str(self.data.iloc[idx]['answer'])

        # Tokenize question and context separately first
        question_tokens = self.tokenizer.tokenize(question)
        context_tokens = self.tokenizer.tokenize(context)
        
        # Truncate question if needed
        if len(question_tokens) > self.max_len_per_segment:
            question_tokens = question_tokens[:self.max_len_per_segment]
            
        # Calculate remaining space for context
        remaining_len = self.max_len - len(question_tokens) - 3  # -3 for [CLS] and two [SEP]
        if len(context_tokens) > remaining_len:
            context_tokens = context_tokens[:remaining_len]

        # Convert back to text
        truncated_question = self.tokenizer.convert_tokens_to_string(question_tokens)
        truncated_context = self.tokenizer.convert_tokens_to_string(context_tokens)

        # Encode with special tokens and padding
        encoding = self.tokenizer(
            truncated_question,
            truncated_context,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
            return_offsets_mapping=True,
            return_token_type_ids=True
        )

        # Find answer span in truncated context
        answer_start = truncated_context.find(answer)
        
        # If answer is not in truncated context, point to [CLS] token
        if answer_start == -1:
            start_position = 0
            end_position = 0
        else:
            answer_end = answer_start + len(answer)
            
            # Convert character positions to token positions
            offset_mapping = encoding.offset_mapping[0].numpy()
            start_position = 0
            end_position = 0
            
            # Find the token positions
            for idx, (start, end) in enumerate(offset_mapping):
                if start != 0 and end != 0:  # Skip special tokens
                    if start <= answer_start <= end:
                        start_position = idx
                    if start <= answer_end <= end:
                        end_position = idx
                        break

        # Ensure positions are valid
        start_position = min(start_position, self.max_len - 1)
        end_position = min(end_position, self.max_len - 1)
        if end_position < start_position:
            end_position = start_position

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'token_type_ids': encoding['token_type_ids'].squeeze(),
            'start_positions': torch.tensor(start_position, dtype=torch.long),
            'end_positions': torch.tensor(end_position, dtype=torch.long)
        }

def train_model(data_path, num_epochs=3, batch_size=8, learning_rate=3e-5, val_split=0.1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    df = pd.read_csv(data_path)
    print(f"Total dataset size: {len(df)}")
    
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained('dbmdz/bert-base-turkish-cased', 
                                                 do_lower_case=False)
    
    # Initialize model
    model = BertForQuestionAnswering.from_pretrained('dbmdz/bert-base-turkish-cased')
    model.to(device)

    # Create dataset with proper max_len
    full_dataset = QADataset(df, tokenizer, max_len=512)  # Explicitly set max_len to 512
    
    # Split dataset
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"Training set size: {train_size}")
    print(f"Validation set size: {val_size}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}')
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_positions = batch['start_positions'].to(device)
            end_positions = batch['end_positions'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                start_positions=start_positions,
                end_positions=end_positions
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1} - Average training loss: {avg_train_loss:.4f}')

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                start_positions = batch['start_positions'].to(device)
                end_positions = batch['end_positions'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    start_positions=start_positions,
                    end_positions=end_positions
                )
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained('./gtu_turkish_qa')
            tokenizer.save_pretrained('./gtu_turkish_qa')
            print("Saved best model checkpoint")

    return model, tokenizer

if __name__ == "__main__":
    try:
        model, tokenizer = train_model('qa_pairs.csv', num_epochs=3)
        print("Training completed successfully!")
    except Exception as e:
        print(f"Error during training: {e}")