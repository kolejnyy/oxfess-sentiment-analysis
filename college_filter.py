from os import listdir


college_dict = {	"Balliol": 	["balliol"],
					"Brasenose": ["brasenose", "bnc"],
					"Christ Church": ["christ church", 'chch'],
					"Corpus Christi": ["corpus christi", 'corpus', 'christi'],
					"Exeter": ["exeter", 'exet'],
					"Harris Manchester": ["harris manchester", 'harris', 'manchester'],
					"Hertford": ["hertford"],
					"Jesus": ["jesus"],
					"Keble": ["keble", 'kebl', 'kbl'],
					"Lady Margaret Hall": ["lady margaret hall", 'lmh'],
					"Lincoln": ["lincoln"],
					"Magdalen": ["magdalen", 'magd'],
					"Mansfield": ["mansfield"],
					"Merton": ["merton"],
					"New": ["new"],
					"Oriel": ["oriel"],
					"Pembroke": ["pembroke", 'pemb'],
					"The Queen's": ["the queen's", "queens", "queenz"],
					"Regent's Park": ["regent's park", 'regents park', 'regents', 'regent'],
					"St Anne's": ["st anne's", "anne's", 'anne'],
					"St Catherine's": ["st catherine's", "catz"],
					"St Edmund Hall": ["st edmund hall", "teddy", "edmund"],
					"St Hilda's": ["st hilda's", 'hilda'],
					"St Hugh's": ["st hugh's", 'hughs', 'hughz', 'hugh'],
					"St John's": ["st john's", "johns", "johnz", "john's", 'john'],
					"St Peter's": ["st peter's", "peters", "peterz", "peter's", 'peter'],
					"Somerville": ["somerville"],
					"Trinity": ["trinity"],
					"University": ["university", "univ"],
					"Wadham": ["wadham", 'wadh'],
					"Worcester": ["worcester", 'wrcs'],
					"Wycliffe Hall": ["wycliffe hall", 'wycliffe'],
					"Union": ["union"],
				}

college_list = list(college_dict.keys())
oxfess_list = listdir('data/oxfess')


def get_oxfesses_for_college(name):
	oxfesses = []
	for oxfess in oxfess_list:
		with open('data/oxfess/{}'.format(oxfess), 'r', encoding='utf-8') as f:
			post = f.readlines()[0].lower()
		for college in college_dict[name]:
			if college in post:
				oxfesses.append(oxfess)
				break
	return oxfesses