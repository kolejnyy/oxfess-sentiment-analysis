import re
from time import sleep


html_path = 'data/htmls/Oxfess_Facebook_190522.html'

with open(html_path, encoding='utf-8') as f:
	html_code = ''.join(f.readlines())

posts = re.findall(r"#[oO]xfess[0-9].{1,1000}", html_code)

print(len(posts))
for i in  range(len(posts)):
	post = posts[i].split("</div>")
	post = [re.sub(r'<.+?>', '', div).replace('&amp;', '&').replace("&gt;", ">").replace("&lt;", '<') for div in post]
	post = re.sub(r'<.+$', '', ' '.join(post))
	post = re.sub(r'Zobacz więcej', '', post)
	if len(post.split()[0])>20:
		print(post)
		sleep(5)
		continue
	with open('data/oxfess/{}.txt'.format(post.split()[0][1:]), 'w', encoding='utf-8') as f:
		f.write(post)
	print(post, '\n')

