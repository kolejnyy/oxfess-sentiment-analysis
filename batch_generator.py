import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn, torch.nn.functional as F, torch.optim as optim



class BatchGenerator:

	def __init__(self) -> None:
		
		with open('data/words.json', 'r') as f:
			self.words = json.load(f)
		self.words = list(self.words.keys())
		self.words = {self.words[i]: i for i in range(len(self.words))}
		self.vocab_size = len(self.words)

		self.data = pd.read_csv('data/cleared_data.csv')
		self.data = self.data.sample(frac=1).reset_index(drop=True)
		
		self.negative_thr = len(self.data)//2
		self.training_ratio = 0.9
		self.training_indices = [i for i in range(len(self.data)) if i < self.negative_thr*self.training_ratio
								or (i >= self.negative_thr and i < self.negative_thr+self.negative_thr*self.training_ratio)]
		
		self.train_lengths = [len(x.split()) for x in self.data.iloc[self.training_indices]['text']]
		self.uniq_train_lengths, self.train_counts = np.unique(self.train_lengths, return_counts=True)
		
		self.length_dict = {self.train_lengths[i]: [] for i in range(len(self.train_lengths))}
		for i in range(len(self.training_indices)):
			self.length_dict[self.train_lengths[i]].append(self.training_indices[i])

		self.batch_prob = np.array([(0 if i==0 else self.train_counts[i]//32) for i in range(len(self.uniq_train_lengths))])
		self.batch_prob = self.batch_prob/np.sum(self.batch_prob)
	
	def generate_batch(self, batch_size=32):
		length = np.random.choice(self.uniq_train_lengths, p=self.batch_prob)
		indices = np.random.choice(self.length_dict[length], batch_size)
		input = torch.Tensor([[self.words[word] for word in self.data.iloc[indices[i]]['text'].split()] for i in range(batch_size)]).long()
		target = torch.Tensor([self.data.iloc[idx]['label'] for idx in indices]).float()
		return input, target