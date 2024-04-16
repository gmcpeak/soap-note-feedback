import os
import re
import pandas as pd
import datetime
from ast import literal_eval  # To convert string representation of lists to actual lists

def get_latest(feedback_path):
    files = [os.path.join(feedback_path, f) for f in os.listdir(feedback_path)
             if os.path.isfile(os.path.join(feedback_path, f)) and 'extracted-tests' in f.lower()]    
    if not files:
        return None
    
    latest = max(files, key=os.path.getctime)
    return latest

df = pd.read_excel(get_latest('../output/'))

def get_highest_feedback_index(dataframe):
    feedback_cols = [col for col in dataframe.columns if col.endswith('_feedback')]
    indices = []
    for col in feedback_cols:
        parts = col.split('_')
        if parts[0].isdigit():
            indices.append(int(parts[0]))
    return max(indices) if indices else None

d = ['_malaria', '_typhoid', '_hiv','_hepatitis', '_urine_analysis', '_anemia']

for w in range(len(d)):
    df['CHEW'+d[w]] = [literal_eval(x)[w] for x in df['SOAP_CHEW1_encoding']]
    df['MO'+d[w]] = [literal_eval(x)[w] for x in df['SOAP_MO_encoding']]

    for j in range(get_highest_feedback_index(df)+1):
        df[str(j)+d[w]] = [literal_eval(x)[w] for x in df[str(j)+'_encoding']]

now = str(datetime.datetime.now())
now = re.sub(r'\W+', '', now)
df.to_excel('../output/tests-expanded-'+now+'.xlsx')
