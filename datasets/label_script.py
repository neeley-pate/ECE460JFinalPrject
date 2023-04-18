import os
import pandas as pd
from pathlib import Path

classes = {'Non-digital Artwork':'Artwork', 'digital':'Digital', 'real-world':'Real World'}

# path for datasets
data_path = os.path.dirname(os.path.realpath(__file__))

for c in classes:
    
    label = classes[c]
    
    cd = os.path.join(data_path, c)
    dirs = os.listdir(cd)
        
    for d in dirs:
        image_gen = Path(os.path.join(cd, d)).iterdir()
        
        results = []

        for i in image_gen:
            i = str(i).split('\\')
            results.append(i[-1])
           
        data = {'image': results, 'label': [label for _ in range(len(results))]}
        df = pd.DataFrame(data)
        df.set_index('image', inplace=True)
        
        df.to_csv(os.path.join(cd,d) + '\\' + d + '.csv')
        
