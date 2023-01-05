from batch_generator import BatchGenerator
from models import LSTM_Analyzer
import torch
from torch import nn, optim
import numpy as np
from time import time

# ==================== Parameters ====================

epochs = 1000
batch_per_ep = 100
interval = 10

exp_name = 'lstm'
load_path = 'weights/lstm/weights_2000.pth'

batch_gen = BatchGenerator()

criterion = nn.MSELoss()
l_r = 0.001

vocab_size 		= batch_gen.vocab_size
embedding_dim 	= 200
hidden_dim 		= 256
n_layers 		= 2
bidirectional 	= False
lstm_dropout 	= 0.3
dropout 		= 0.3

# ===================== Model ======================

analyzer = LSTM_Analyzer(bidirectional=bidirectional,
						 vocab_size=vocab_size,
						 embedding_dim=embedding_dim,
						 hidden_dim=hidden_dim,
						 n_layers=n_layers,
						 lstm_dropout=lstm_dropout,
						 dropout=dropout)
if load_path is not None:
	analyzer.load_state_dict(torch.load(load_path))

optimizer = optim.Adam(analyzer.parameters(), lr=l_r)

# ==================== Training ====================

for i in range(epochs):

	epoch_loss = 0
	start_time = time()
	
	for j in range(batch_per_ep):
		input, target = batch_gen.generate_batch()
		output, hs = analyzer(input)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()
		epoch_loss += loss.item()
	
	epoch_loss /= batch_per_ep
	print("Epoch {}/{}:\tloss = {:.4f}\t   epoch_time = {:.2f} [s]".format(i+1, epochs, epoch_loss, time()-start_time))
	
	if i%interval == interval-1:
		torch.save(analyzer.state_dict(), 'weights/{}/weights_{}.pth'.format(exp_name, (i+1)*batch_per_ep))
