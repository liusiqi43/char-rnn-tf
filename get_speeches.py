from bs4 import BeautifulSoup
from urls import urls
from goose import Goose
import pickle

result = []

g = Goose()

for u in urls:
    url = 'https://www.whitehouse.gov%s' % u
    article = g.extract(url=url)
    result.append(article.cleaned_text)

with open('speeches.pkl', 'wb') as f:
    pickle.dump(result, f)
