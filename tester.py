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
from models import LSTM_Analyzer

batch_ep = 10000
interval = 100

exp_name = 'lstm'
load_path = 'weights/lstm/weights_500.pth'

analyzer = LSTM_Analyzer(bidirectional=True)
batch_gen = BatchGenerator()
criterion = nn.MSELoss()
optimizer = optim.Adam(analyzer.parameters(), lr=0.001)

analyzer.load_state_dict(torch.load(load_path))

last_losses = [1]*50

for i in range(batch_ep):
	input, target = batch_gen.generate_batch()
	output, hs = analyzer(input)
	loss = criterion(output, target)
	loss.backward()
	optimizer.step()
	optimizer.zero_grad()
	last_losses = last_losses[1:] + [loss.item()]
	print("Batch {}/{}:\tloss = {:.4f}\t    avg50_loss = {:.4f}".format(i+1, batch_ep, loss.item(), np.mean(last_losses)))
	
	if i%interval == interval-1:
		torch.save(analyzer.state_dict(), 'weights/{}/weights_{}.pth'.format(exp_name, i+1))
		print("Saved weights")
