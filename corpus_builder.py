from preprocessing import preprocess_post
import pandas as pd
import numpy as np
import json

def build_data(input_data, save_path):

	word_count = {}
	data = pd.read_csv(input_data, encoding='latin-1', header=None)
	data = data.dropna()
	data = data.reset_index(drop=True)
	
	new_data = {'text': [], 'label': []}
	new_data['text'] = data[5].apply(preprocess_post)
	new_data['label'] = data[0].apply(lambda x: 1 if x == 4 else -1)
	new_data = pd.DataFrame(new_data)

	for x in new_data['text']:
		for word in x.split():
			if word in word_count:
				word_count[word] += 1
			else:
				word_count[word] = 1
	word_count_keys = list(word_count.keys())
	for x in word_count_keys:
		if word_count[x] < 50:
			del word_count[x]

	y = []
	for x in new_data['text']:
		y.append(' '.join([z for z in x.split() if z in word_count]))
	new_data['text'] = y
	new_data = new_data[new_data['text'] != '']
	new_data = new_data.reset_index(drop=True)

	new_data.to_csv(save_path+'/cleared_data.csv', index=False)
	print('Saved cleared data to {}'.format(save_path+'/cleared_data.csv'))
	with open(save_path+'/words.json', 'w') as f:
		json.dump(word_count, f)
	print('Words saved to {}'.format(save_path+'/words.json'))