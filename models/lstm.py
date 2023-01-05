import torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim
import numpy as np


class LSTM_Analyzer(nn.Module):

	def __init__(self, vocab_size=12362, embedding_dim=200, hidden_dim=256, n_layers=2, bidirectional=False, lstm_dropout=0.3, dropout=0.3):

		super().__init__()

		self.vocab_size = vocab_size
		self.hidden_dim = hidden_dim
		self.n_layers 	= n_layers
		self.embedding_dim = embedding_dim
		self.bidirectional = bidirectional

		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=lstm_dropout, batch_first=True)
		self.fc = nn.Linear(2*hidden_dim if self.bidirectional else hidden_dim, 1)
		
		self.dropout = nn.Dropout(dropout)
		self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		embedded = self.embedding(x)
		output, (hidden_state, cell_state) = self.lstm(embedded)
		output = self.dropout(output)
		imp_out = output[:,-1,:].squeeze(1)
		if self.bidirectional:
			imp_out = torch.concat((output[:,-1,:self.hidden_dim].squeeze(1),output[:,0,self.hidden_dim:].squeeze(1)), dim=1)
		predictions = self.fc(imp_out)
		return self.sigmoid(predictions).squeeze(1), (hidden_state, cell_state)