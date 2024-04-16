test_extract_prompt = \
"""
Below is a document which discusses an interaction between a community health worker and their patient. Find the #ADDITIONALTESTS section and answer the following question about the document:
 
What additional tests are recommended by the document?
"""

test_extract_prompt_2 = \
"""
Based on you previous answer, I want you to identify which of the following tests you are recommending:
[Malaria RDT, Typhoid RDT, HIV, Hepatitis, Urine analysis, Anemia]

Do this by creating a Python list of one-hot encodings, where 1 indicates the test is being recommended, and 0 indicates the test is not being recommended.
You should recognize things like "MP" being a Malaria test, "Widal" and "Salmonella" both being Typhoid tests, and "PCV" being for Anemia.

For example, if you recommend tests for Typhoid, Hepatitis, and a Urine analysis, then your output should be:
[0, 1, 0, 1, 1, 0]

If you recommend tests for HIV, Malaria, and an Anemia test then the output should be:
[1, 0, 1, 0, 0, 1]

If you did not provide any test recommendations, then it should be:
[0, 0, 0, 0, 0, 0]

Answer with your Python list and ONLY your Python list. ONLY count tests that have been explicitly recommended.
"""

import re
import os
import math
import datetime
import pandas as pd
from tqdm import tqdm
from joblib import Parallel,delayed 
from openai import OpenAI
import sys
path_to_key = os.path.abspath('../assets')
if path_to_key not in sys.path:
    sys.path.append(path_to_key)
import key

client = OpenAI(
    api_key = key.api_key # Your API key goes here
)

def get_latest(feedback_path):
    files = [os.path.join(feedback_path, f) for f in os.listdir(feedback_path)
             if os.path.isfile(os.path.join(feedback_path, f)) and 'feedback' in f.lower()]
    if not files:
        return None
    latest = max(files, key=os.path.getctime)
    return latest

feedback_df = pd.read_excel(get_latest('../output/'))

def predict(sys_prompt_1, sys_prompt_2, target):
    response = client.chat.completions.create(
        model="gpt-4-0125-preview", # GPT-4 Turbo
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": sys_prompt_1 + target},
        ],
    )
    temp = response.choices[0].message.content
    response = client.chat.completions.create(
        model="gpt-4-0125-preview", # GPT-4 Turbo
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": sys_prompt_1 + target},
            {"role": "assistant", "content": temp},
            {"role": "system", "content": sys_prompt_2},
        ],
    )
    out = response.choices[0].message.content
    return out

now = str(datetime.datetime.now())
now = re.sub(r'\W+', '', now)

out_df = pd.DataFrame()

def get_encodings(col):
    t = []
    for j in tqdm(range(len(feedback_df.assessment_id))):
        try:
            s = predict(test_extract_prompt, test_extract_prompt_2, col[j])
            t.append(s)
        except Exception as e:
            print('ERROR! Retrying...')
            print(e)
    return t

def process(i):
    out_df[str(i)+'_feedback'] = feedback_df[str(i)+'_feedback']
    out_df[str(i)+'_encoding'] = get_encodings(feedback_df[str(i)+'_feedback'])

def get_highest_feedback_index(dataframe):
    feedback_cols = [col for col in dataframe.columns if col.endswith('_feedback')]
    indices = []
    for col in feedback_cols:
        parts = col.split('_')
        if parts[0].isdigit():
            indices.append(int(parts[0]))
    return max(indices) if indices else None

Parallel(n_jobs=-1, backend='threading')(delayed(process)(i) for i in range(get_highest_feedback_index(feedback_df)+1))
out_df['assessment_id'] = feedback_df.assessment_id
out_df['SOAP_CHEW1'] = feedback_df.SOAP_CHEW1
out_df['SOAP_CHEW1_encoding'] = get_encodings(feedback_df.SOAP_CHEW1)
out_df['SOAP_CHEW2'] = feedback_df.SOAP_CHEW2
out_df['SOAP_CHEW2_encoding'] = get_encodings(feedback_df.SOAP_CHEW2)
out_df['SOAP_MO'] = feedback_df.SOAP_MO
out_df['SOAP_MO_encoding'] = get_encodings(feedback_df.SOAP_MO)
out_df.to_excel('../output/extracted-tests-'+now+'.xlsx')
