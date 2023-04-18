from requests import get
from pandas import read_csv
from os.path import exists, dirname, join
import random

df = read_csv('diffusion_prompts.csv')
directory = join(dirname(dirname(__file__)), 'digital/StableDiffusion')
random.seed(42)
i = 0
while i < 5000:
    index = random.randint(0, df.shape[0])
    row = df.iloc[index]
    path = ''
    if row.loc['source_site'] == 'stablediffusionweb.com':
        fileName = row.loc['id'] + '.png'
        path = join(directory, fileName)
    else:
        fileName = row.loc['id'] + '.jpg'
        path = join(directory, fileName)
    print('Path: ', path)
    if not exists(path):
        response = get(row.loc['url'], stream = True)
        if response.ok:
            with open(path, 'wb') as f:
                f.write(response.content)
        else:
            i = i - 1
    i = i + 1
