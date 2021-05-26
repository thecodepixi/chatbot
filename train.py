import json 
from nltk_utils import *
import numpy as np

import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import NeuralNet

with open('intents.json', 'r') as f: 
  intents = json.load(f)
  
  all_words = []
  tags = []
  xy = []

  for intent in intents['intents']: 
    tag = intent['tag']
    if tag not in tags: 
      tags.append(tag) 
    for pattern in intent['patterns']:
      words = tokenize(pattern)
      all_words.extend(words)
      xy.append((words, tag))

  ignore_words = ['?', '.', '!', ',']
  all_words = [stem(w) for w in all_words if w not in ignore_words]
  all_words = sorted(all_words)
  tags = sorted(tags)

X_train = []
y_train = []

for (pattern, tag) in xy:
  bag = bag_of_words(pattern, all_words)
  X_train.append(bag)

  label = tags.index(tag)
  y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):

  def __init__(self):
    self.n_samples = len(X_train)
    self.x_data = X_train
    self.y_data = y_train

  def __getitem__(self, index):
    return self.x_data[index], self.y_data[index]

  def __len__(self):
    return self.n_samples

batch_size = 10
hidden_size = 10
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = .003
num_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

device = torch.device('cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
for epoch in range(num_epochs):
    for (words, labels) in train_loader:
      words = words.to(device)
      labels = labels.to(dtype=torch.long).to(device)
      
      # Forward pass
      outputs = model(words)
      loss = criterion(outputs, labels)
      
      # Backward and optimize
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
        
    if (epoch+1) % 100 == 0:
      print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


print(f'final loss: {loss.item():.4f}')

data = {
  "model_state": model.state_dict(),
  "input_size": input_size,
  "output_size": output_size,
  "hidden_size": hidden_size,
  "all_words": all_words,
  "tags": tags
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'Training complete. Filed saved to {FILE}.')