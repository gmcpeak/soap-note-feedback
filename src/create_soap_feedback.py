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

prompts_df = pd.read_excel('../input/prompts.xlsx')
notes_df = pd.read_excel('../input/soap-notes.xlsx')

client = OpenAI(
    api_key = key.api_key # Your API key goes here
)

def predict(sys_prompt, target):
    response = client.chat.completions.create(
        model="gpt-4-0125-preview", # GPT-4 Turbo
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": sys_prompt + target},
        ],
    )
    out = response.choices[0].message.content
    return out 

now = str(datetime.datetime.now())
now = re.sub(r'\W+', '', now)

out_df = pd.DataFrame()

def process(i):
    if prompts_df['flag'][i] == 1:
        t = []
        sys_prompt = str(prompts_df['prompt'][i])
        for j in tqdm(range(len(notes_df.assessment_id))):
            try:
                s = predict(sys_prompt, notes_df.SOAP_CHEW1[j])
                t.append(s)
            except Exception as e:
                print('ERROR! Retrying...')
                print(e)
        out_df[str(i)+'_feedback'] = t

Parallel(n_jobs=-1, backend='threading')(delayed(process)(i) for i in range(len(prompts_df.prompt)))
out_df['assessment_id'] = notes_df.assessment_id
out_df['SOAP_CHEW1'] = notes_df.SOAP_CHEW1
out_df['SOAP_CHEW2'] = notes_df.SOAP_CHEW2
out_df['SOAP_MO'] = notes_df.SOAP_MO
out_df.to_excel('../output/feedback-'+now+'.xlsx')
