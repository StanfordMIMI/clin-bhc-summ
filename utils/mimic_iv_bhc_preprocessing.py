import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt

df = pd.read_csv("mimic-iv-note.csv")
df['text'] = df.text + '\n' + df.discharge_instructions
df['text'] = df['text'].astype(str)
df['bhc'] = df['BHC'].astype(str)
df = df[['index', 'text', 'bhc']]

df = df.reset_index(drop=True)

df['text'] = df['text'].apply(lambda x: re.sub(' +', ' ', x))
df['bhc'] = df['bhc'].apply(lambda x: re.sub(' +', ' ', x))

df = df[df['text'].apply(lambda x: x[88:91] == 'Sex') == True]
df['text'] = df['text'].apply(lambda x: x[88:])

categories = ["Sex:", "Service:", "Allergies:", "Attending:", "Chief Complaint:", "Major Surgical or Invasive Procedure:", 
 "History of Present Illness:", "Past Medical History:", "Social History:", "Family History:", "Physical Exam:",
 "Pertinent Results:", "Medications on Admission:", "Discharge Medications:", "Discharge Disposition:",
 "Discharge Diagnosis:", "Discharge Condition:", "Followup Instructions:", "Discharge Instructions:"]

categories_upper = ['<' + i[:-1].upper() + '>' for i in categories]

categories = ["Sex:", "Service:", "Allergies:", "Attending:", "Chief Complaint:", "Major Surgical or Invasive Procedure:", 
 "History of Present Illness:", "Past Medical History:", "Social History:", "Family History:", "Physical Exam:",
 "Pertinent Results:", "Medications on Admission:", "Discharge Medications:", "Discharge Disposition:",
 "Discharge Diagnosis:", "Discharge Condition:", "Followup Instructions:", "Discharge Instructions:"]

for i in range(len(categories)):
    df['text'] = df['text'].apply(lambda x: x.replace(categories[i], categories_upper[i]))
    
df['text'] = df['text'].apply(lambda x: x.replace('\n', " "))
df['text'] = df['text'].apply(lambda x: x.replace('=================', '-'))
df['text'] = df['text'].apply(lambda x: x.replace('=============', '-'))
df['text'] = df['text'].apply(lambda x: x.replace(' ###', '###'))
df['text'] = df['text'].apply(lambda x: x.replace(': ', ':'))
df['text'] = df['text'].apply(lambda x: x.replace('###', ' ### '))
df['text'] = df['text'].apply(lambda x: x.replace(':', ': '))
df['text'] = df['text'].apply(lambda x: re.sub('\s+', ' ', x))
df['text'] = df['text'].apply(lambda x: x[1:])
df['text'] = df['text'].apply(lambda x: x.replace('\x95', '-'))

df['bhc'] = df['bhc'].apply(lambda x: x.replace("Brief Hospital Course:\n", ""))
df['bhc'] = df['bhc'].apply(lambda x: x.replace("BRIEF HOSPITAL COURSE:\n", ""))
df['bhc'] = df['bhc'].apply(lambda x: x.replace("BRIEF HOSPITAL COURSE", ""))
df['bhc'] = df['bhc'].apply(lambda x: x.replace("Brief Hospital Course", ""))
df['bhc'] = df['bhc'].apply(lambda x: x.replace("\n", " "))
df['bhc'] = df['bhc'].apply(lambda x: re.sub('\s+', ' ', x))

df = df.reset_index(drop=True)

df.to_csv("mimic-iv_preprocessed.csv")