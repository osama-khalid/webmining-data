import csv

data = []
with open('urdudigest.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        data.append(row)

sentences = []
for url in data:
    sentences.append(''.join(url))

import random
random.shuffle(sentences)

sentences = sentences[:1000]

#!pip install transformers
#!pip install torch

import torch
from torch.utils.data import DataLoader
from transformers import BertForSequenceClassification,BertTokenizer, BertForMaskedLM, AdamW
from transformers import BertTokenizer, BertModel, BertConfig

# Load the pre-trained model and tokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model = BertModel.from_pretrained('bert-base-multilingual-cased')

tokenized_sentences = tokenizer.batch_encode_plus(
    sentences,
    add_special_tokens=True,
    padding='longest',
    truncation=True,
    return_tensors='pt'
)

# Prepare the input tensors
input_ids = tokenized_sentences['input_ids']
attention_mask = tokenized_sentences['attention_mask']

# Set the batch size
batch_size = 2

import tqdm
# Create data loader for batch processing
data = torch.utils.data.TensorDataset(input_ids, attention_mask)
data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size)

# Set the model to training mode
model.train()

# Fine-tuning loop
for batch in tqdm.tqdm(data_loader):
    batch_input_ids, batch_attention_mask = batch

    # Clear gradients
    model.zero_grad()

    # Forward pass
    outputs = model(
        input_ids=batch_input_ids,
        attention_mask=batch_attention_mask
    )

model.save_pretrained('fine-tuned-model')
tokenizer.save_pretrained('fine-tuned-model')
# Test the model
model.eval()



tokenizer_mult = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
model_mult = BertModel.from_pretrained('bert-base-multilingual-cased', config=config)

def extract_cls_token(sentence,tokenizer,model):
    # Tokenize the input sentence
    tokens = tokenizer.tokenize(sentence)

    # Add the special tokens [CLS] and [SEP]
    tokens = ['[CLS]'] + tokens + ['[SEP]']

    # Convert the tokens to token IDs
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Convert the input IDs to a PyTorch tensor
    input_tensor = torch.tensor([input_ids])

    # Get the hidden states from the BERT model
    with torch.no_grad():
        outputs = model(input_tensor)

    # Extract the hidden states of the CLS token (the first token)
    cls_token = outputs.last_hidden_state[0][0]

    return cls_token

text_set = []
senses = ['vision','auditory','interoceptive','haptic','gustatory','smell']
for sense in senses:
    file = open(sense+'.txt','r').read().split('\n')

    for row in file:
        text_set.append(row)

# Import required libraries
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import tqdm
from sklearn.metrics import f1_score
import pandas as pd

import csv
import requests
train_texts = []
train_labels = []
label_idx = {}

i = 0
LABELS = ['V','A','I','H','G','O']
file = open('vision.txt','r').read().split('\n')

for row in file:
    train_texts.append(row.lower())
    if 0 not in label_idx:
        label_idx[0] = 0
    train_labels.append(label_idx[0])  

file = open('auditory.txt','r').read().split('\n')

for row in file:
    train_texts.append(row.lower())
    if 1 not in label_idx:
        label_idx[1] = 1
    train_labels.append(label_idx[1])   

file = open('interoceptive.txt','r').read().split('\n')

for row in file:
    train_texts.append(row.lower())
    if 2 not in label_idx:
        label_idx[2] = 2
    train_labels.append(label_idx[2])  

file = open('haptic.txt','r').read().split('\n')

for row in file:
    train_texts.append(row.lower())
    if 3 not in label_idx:
        label_idx[3] = 3
    train_labels.append(label_idx[3])  

file = open('gustatory.txt','r').read().split('\n')

for row in file:
    train_texts.append(row.lower())
    if 4 not in label_idx:
        label_idx[4] = 4
    train_labels.append(label_idx[4])  

file = open('smell.txt','r').read().split('\n')

for row in file:
    train_texts.append(row.lower())
    if 5 not in label_idx:
        label_idx[5] = 5
    train_labels.append(label_idx[5])  

# Load dataset
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoded_input = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'text': text,
            'input_ids': encoded_input['input_ids'].flatten(),
            'attention_mask': encoded_input['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
# Define hyperparameters
MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 2e-5

num_labels = len(label_idx)


# Tokenize input texts
tokenizer_mult = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
dataset = CustomDataset(train_texts, train_labels, tokenizer_mult, MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Load BERT model for classification task
config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True, num_labels = num_labels)
model_mult = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', config=config)
# Load BERT model for classification task
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model_mult.to(device)


# Define optimizer and loss function
optimizer = AdamW(model_mult.parameters(), lr=LEARNING_RATE, eps=1e-8)
loss_fn = torch.nn.CrossEntropyLoss()



# Train the model
for epoch in range(EPOCHS):
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        outputs = model_mult(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs[0]
        
        # Backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Evaluate model after each epoch
    y_true = labels.cpu().detach().numpy()
    y_pred = torch.argmax(outputs[1], dim=1).cpu().detach().numpy()
    f1 = f1_score(y_true, y_pred,average='micro')
    print('Epoch:', epoch+1, 'F1 score:', f1)

# Save the model
#torch.save(model.state_dict(), 'model.pt')
model_mult.save_pretrained('saved_model')
tokenizer_mult.save_pretrained('saved_model')
# Test the model
model_mult.eval()    


def extract_cls_token_(sentence,tokenizer,model):
    # Tokenize the sentence
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids']
    attention_mask = encoding['attention_mask']

    # Extract hidden states from the first layer
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states

    first_layer_hidden_states = hidden_states[0]

    # Access the representation of the CLS token
    cls_representation = first_layer_hidden_states[:, 0, :]
    return(cls_representation)

import numpy as np
import tqdm
text = "she saw a dog"
cls_base = []
cls_mult = []
for text in tqdm.tqdm(text_set):

    cls_mult.append(np.array(extract_cls_token_(text,tokenizer_mult,model_mult)))
    cls_base.append(np.array(extract_cls_token(text,tokenizer,model)))  


print(cls_base[0][0],cls_mult[0][0])
import numpy as np
def procrustes_transform_(A, B):
    # Convert A and B to NumPy arrays
    A = np.array(A)
    B = np.array(B)

    # Apply Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(np.dot(B.T, A))

    # Calculate the transformation matrix
    transformation_matrix = np.dot(U, Vt)
    return(transformation_matrix)


R = procrustes_transform_(cls_base, cls_mult)


import torch
import torch.nn.functional as F

def predict_base(texts, model, tokenizer):
    # Tokenize the input texts
    encoded_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

    # Convert the input tokens to direct 1x768 representations
    input_vectors = []
    for input_id in encoded_inputs['input_ids']:
        input_vector = model.bert.embeddings.word_embeddings(input_id)
        
        # Perform element-wise multiplication with B
        
        input_vectors.append(input_vector)
    
    # Stack the input vectors into a tensor
    input_vectors = torch.stack(input_vectors, dim=0)
    
    # Create attention mask tensor
    attention_mask = encoded_inputs['attention_mask']
    
    # Move the input tensors to the GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_vectors = input_vectors.to(device)
    attention_mask = attention_mask.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make predictions on the input vectors
    with torch.no_grad():
        outputs = model(inputs_embeds=input_vectors, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels = F.softmax(logits, dim=-1)
    
    return predicted_labels

import torch
import torch.nn.functional as F

def predict(texts, model, tokenizer, B,model_mult):
    # Tokenize the input texts
    encoded_inputs = tokenizer(texts, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    
    # Convert the input tokens to direct 1x768 representations
    input_vectors = []
    for input_id in encoded_inputs['input_ids']:
        input_vector = model_mult.embeddings.word_embeddings(input_id)
        transformed_A=[]
        for R in B:
          transformed_batch_A_np = torch.matmul(input_vector, torch.tensor(R))

          # Convert the transformed array back to a PyTorch tensor
          transformed_batch_A = transformed_batch_A_np

          # Reshape the transformed tensor to its original shape
          transformed_batch_A = transformed_batch_A.view(input_vector.size(0), input_vector.size(1), -1)

          # Append the transformed batch to the result
          transformed_A.append(transformed_batch_A)

        # Concatenate the transformed batches
        input_vector = torch.cat(transformed_A, dim=0)
        #input_vector = torch.matmul(input_vector, torch.tensor(B))
        input_vectors.append(input_vector)
        
    
    # Stack the input vectors into a tensor
    input_vectors = torch.stack(input_vectors, dim=0)
    
    # Create attention mask tensor
    attention_mask = encoded_inputs['attention_mask']
    
    # Move the input tensors to the GPU (if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_vectors = input_vectors.to(device)
    attention_mask = attention_mask.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make predictions on the input vectors
    with torch.no_grad():
        outputs = model(inputs_embeds=input_vectors, attention_mask=attention_mask)
        logits = outputs.logits
        predicted_labels = torch.argmax(logits, dim=1)
        predicted_labels = F.softmax(logits, dim=-1)
    
    return predicted_labels.tolist()


texts = 'The butterfly gracefully fluttered from flower to flower, its delicate wings adding a touch of beauty to the garden.'

X=predict(texts.lower(),model_mult,tokenizer_mult,R,model)
Y=predict_base(texts.lower(),model_mult,tokenizer_mult)
print(texts)
print(X)
print(Y)        

texts = "میں نے چوہے کی آواز سنی"

X=predict(texts.lower(),model_mult,tokenizer_mult,R,model)
Y=predict_base(texts.lower(),model_mult,tokenizer_mult)
print(texts)
print(X)
print(Y)        
