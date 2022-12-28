from bs4 import BeautifulSoup as bs
from urllib.request import (
    urlopen, urlparse, urlunparse, urlretrieve)
import os
import sys
import re
from time import sleep, time
from os import listdir
import cohere
from cohere.classify import Example
import numpy as np
import college_filter as clf
from tqdm import tqdm
import pandas as pd
from corpus_builder import build_data
from batch_generator import BatchGenerator
import torch
from torch import nn, functional as F, optim

class NNet(nn.Module):
	def __init__(self) -> None:
		super().__init__()

		self.fc1 = nn.Linear(8, 128)
		self.fc2 = nn.Linear(128, 512)
		self.fc3 = nn.Linear(512, 512)
		self.fc4 = nn.Linear(512, 1)

		self.sigmoid = nn.Sigmoid()
	
	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = F.relu(self.fc3(x))
		x = self.sigmoid(self.fc4(x))
		return x


model = NNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

batches = []
epochs = 10

for i in range(epochs):
	for input, target in batches:
		optimizer.zero_grad()
		output = model(input)
		loss = loss_fn(output, target)
		loss.backward()
		optimizer.step()