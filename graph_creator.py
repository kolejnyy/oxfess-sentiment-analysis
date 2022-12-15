import matplotlib.pyplot as plt
from cohere_ranking import result_dict
from college_filter import get_oxfesses_for_college
import pandas as pd

def prepare_data(res_dict):
	# Prepare data for graph
	data = []
	for college in res_dict:
		if len(get_oxfesses_for_college(college)) >= 10:
			data.append((res_dict[college], college))
	data.sort()
	college_list = []
	score_list = []
	for college in data:
		college_list.append(college[1])
		score_list.append(college[0])
	data_dict = {'college': college_list, 'score': score_list}
	return pd.DataFrame(data_dict)

data = prepare_data(result_dict)
data.plot(	x = 'college', y = 'score',
			kind='barh',
			figsize=(14, 10),
			color=(data['score'] > 0).map({True: 'g',
                                            False: 'r'}))
plt.title('College Sentiment Ranking - cohere.ai')
plt.xlim(-1, 1)
plt.legend().remove()
plt.savefig('images/cohere_ranking.png')
plt.show()