from bs4 import BeautifulSoup as bs
from urllib.request import (
    urlopen, urlparse, urlunparse, urlretrieve)
import os
import sys
import re
from time import sleep
from os import listdir
import cohere
from cohere.classify import Example
import numpy as np
import college_filter as clf
from tqdm import tqdm

#co = cohere.Client('{apikey}')

examples = [	Example("i think we can all agree that the cinderella girl at ball stole the show", "Positive Review"),
				Example("How closely do they check scholars gowns? I barely passed prelims - can I just buy and wear one for the flex???", "Neutral Review"),
				Example("which societies have the best parties?", "Positive Review"),
				Example("I feel sick after sitting in the chamber for so long.. this was so clearly a witch-hunt against a person of colour orchestrated by members of the same slate who stood huddled up together like a gang of Tory penguins. Go home and touch some grass.", "Negative Review"),
				Example("sorry but third years have absolutely rancid vibes", "Negative Review"),
				Example("Does anyone know where I can get my MacBook battery serviced in Oxford? Thanks!", "Neutral Review"),
				Example("I’m sick and tired of sitting in tutorials with private school knobs who have absolutely no issues with chatting absolute bollocks. I’m getting nothing out of the classes because 1. I don’t have any of the background knowledge that the others apparently do so I don’t even understand most of the content without having to an insane amount of extra reading on top of the already mental workload", "Negative Review"),
				Example("Oxwow to grease @ warwick", "Positive Review"),
				Example("Actually impressive that STR is more controversial than basilford college instagram right now. Satire is dead", "Positive Review"),
				Example("Oxlove to Ahmed's. The messiah has returned!!!!!!!!", "Positive Review"),
				Example("oxlove to the guy wearing clothes in Oxford today  saw you and thought you looked amazing. coffee?", "Positive Review"),
				Example(" Is it just me or has there basically been nothing on for 5th week? Like there are hardly any more welfare events than usual", "Negative Review"),
				Example("College kids PLEASE contact your college parents, we're excited to meet you! xx", "Positive Review"),
				Example("oxlove to Lincoln Lane, we had enough pizza at the JCR meeting to pass that motion", "Positive Review"),
				Example("Oxlove to the Vogue article @ the ox blue. It was refreshing to have a critical opinion on something that is often brushed off yet is so damaging to art.", "Positive Review"),
				Example("Oxhate to Oxford Opera Society.", "Negative Review"),
				Example("how can winter ball seriously be 215 quid?? for that money i want my own heater", "Negative Review"),
				Example("Oxshoutout to the random fresher who stopped me in week 8 randomly to have a chat because she remembered me from helping out with fresher's week. Made my day seeing how your year are getting on.", "Positive Review"),
				Example("I don’t care who is the President, the true face of the Oxford Union is the skinny guy that asks you not to take photos of the guests.", "Negative Review"),
				Example("Oxhate to trying to escape your degree by going to a different continent and still bumping into someone you have a class with", "Negative Review"),
				Example("Hot take - the oxford college marriages system is just queer platonic partners for straight people", "Neutral Review"),
				Example("As a student can we actually visit any college we like for free? Do we still have to tell the porters? Can we have guests?", "Neutral Review"),
				Example("Why is it always the same people getting parts in oxford drama? It makes it impossible to get involved as a newbie/someone with little drama experience", "Negative Review"),
				Example("when will oulc learn that being inclusive doesn’t mean being dull…also, the antisemitism at these so called inclusive events", "Negative Review"),
				Example("Oxhate to the biochem department for setting a collection with just one weeks notice and not even saying what they are testing us on", "Negative Review"),
				Example("Oxhate to St Hugh’s College JCR for their blatant sexist treatment of the only worthy candidate of JCR presidency. Shameless that we still treat women who can assert themselves and actually have a voice this way when we as a college were founded on the basis of educating women and empowering their status as polictical and academic thinkers.", "Negative Review"),
				Example("Late oxlove for Cambridge pool team you guys are great", "Positive Review"),
				Example("ngl those school kids deserved better.  imagine going to a school that hasn’t sent people to oxford and you get to go see an oxford union debate and are excited but instead all you see is the bs maneuvering to get rid of the president", "Negative Review"),
				Example("can people start shitposting union drama again pls", "Neutral Review"),
				Example("Day #5 of posting the Bee Movie Script: laws", "Neutral Review"),
				Example("Second year chemists and above pls help! How do you revise organic chem effectively for prelims? I have no idea how to learn all these specific mechanisms!", "Neutral Review"),
				Example("Geography finalists, I’m so overwhelmed with extended essays, I haven’t finished any of them, I’ve not started one and I’m really starting to panic - please tell me I’m not alone…", "Negative Review"),
				Example("Oxsad to not having any bops in the JCR @J this term. Club bops just don't hit the same", "Negative Review"),
				Example("Greeting oxfess, Soloman here. To the boy in the MIT puffer. You welcome to Soloman’s Grill any time brother", "Positive Review"),
				Example("Wow I look tired? Really? I had no idea. Thank you for telling me. You’re so insightful! Seems like you’re super smart, have you thought about applying to Oxford?", "Negative Review"),
				Example(" Deliveroo ‘cyclists’ don’t know the rules of the road and have clearly just been given E bikes and told to get going.", "Negative Review"),
				Example("It's outrageous that ChCh fine students £50 if they don't check out by 9.30am. Not everyone has parents an hour's drive away", "Negative Review"),
				Example("Oxwtf to the person wearing a backpack at musicals night at bully tonight", "Neutral Review"),
				Example("The absence of mccoys is starting to get to me. This is a violation of Rights of An Oxford Student.", "Negative Review"),
				Example("Union RO world as McDonald’s items:   AC - mayo chicken  KD - McFlurry", "Neutral Review"),
				Example("Best college for postgrad accommodation?", "Neutral Review"),
				Example(" oxford colleges as love island contestants:  hertford : danika jesus: davide  exeter: ekin-su", "Neutral Review"),
				Example("Oxford colleges and whether Maggie Thatcher was an alumna", "Neutral Review"),
			]


decode = {"Negative Review": -1, "Neutral Review": 0, "Positive Review": 1}

college_list = clf.college_list


def get_college_score(college_name):
	oxfess_list = clf.get_oxfesses_for_college(college_name)
	result = 0
	count = 0

	for x in tqdm(oxfess_list):
		with open('data/oxfess/' + x, 'r', encoding='utf-8') as f:
			post = f.readlines()[0]
		inputs = [post]
		
		response = co.classify(
			model='large',
			inputs=inputs,
			examples=examples)

		for cls in response.classifications:
			pred = cls.prediction
			conf = cls.confidence
			pred = decode[pred]
			result += pred*conf
		
		count += 1
		sleep(0.3)

	if count == 0:
		return None

	return result/count

result_dict =  {'Balliol': 0.36005022464285713,
				'Brasenose': -0.26371100416666665,
				'Christ Church': -0.03819371894736843,
				'Corpus Christi': -0.097097245,
				'Exeter': 0.01733342866666668,
				'Harris Manchester': 0.8262625,
				'Hertford': -0.4678783864705882,
				'Jesus': 0.3872872528571428,
				'Keble': -0.5258648624,
				'Lady Margaret Hall': 0.235525796,
				'Lincoln': 0.4510041727272727,
				'Magdalen': 0.4212419968181818,
				'Mansfield': 0.12947384000000003,
				'Merton': 0.22004988408163267,
				'New': -0.08031171761904757,
				'Oriel': 0.09716546000000001,
				'Pembroke': -0.215155299,
				"The Queen's": 0.29484134727272726,
				"Regent's Park": -0.037863236666666654,
				"St Anne's": -0.3266609471428571,
				"St Catherine's": 0.03497902235294119,
				'St Edmund Hall': -0.04138604956521739,
				"St Hilda's": 0.21295472000000001,
				"St Hugh's": -0.06360321899999999,
				"St John's": -0.03872522555555553,
				"St Peter's": -0.2509757321428571,
				'Somerville': -0.05496264333333337,
				'Trinity': -0.16260876166666668,
				'University': -0.36395036517241375,
				'Wadham': 0.43760745736842105,
				'Worcester': 0.2342276753846154,
				'Union': -0.4667067216058394}

# for college_name in college_list:
# 	score = get_college_score(college_name)
# 	print(college_name, score)
# 	result_dict[college_name] = score


# print("\n\n\n", result_dict)