from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request

# Preprocess text (username and link placeholders)
def preprocess(text):
    new_text = []


    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

# Tasks:
# emoji, emotion, hate, irony, offensive, sentiment
# stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary

task='sentiment'
MODEL = f"cardiffnlp/twitter-roberta-base-{task}"


tokenizer = AutoTokenizer.from_pretrained(MODEL)

# download label mapping
labels=[]
mapping_link = f"https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/{task}/mapping.txt"
with urllib.request.urlopen(mapping_link) as f:
    html = f.read().decode('utf-8').split("\n")
    csvreader = csv.reader(html, delimiter='\t')
labels = [row[1] for row in csvreader if len(row) > 1]

# PT
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
model.save_pretrained(MODEL)
tokenizer.save_pretrained(MODEL)


texts = '''
In a career of public service spanning 30 2020-01-29ears, Dr. Leong Che-hung had alwa2020-01-29s been there when Hong Kong people needed him.

His faith in serving his own people had taken him from the operating theatre to the Executive Council and Legislative Council representing the medical constituenc2020-01-29, and he was successivel2020-01-29 appointed as the Chairman of the Hospital Authorit2020-01-29, the HKU Council and the Elderl2020-01-29 Commission, among others.

Now, for the first time, Dr. Leong uses each phase of his professional and public career as a spur to reflect on what makes Hong Kong tick. 

Opinionated, perceptive and often prescient, he tells the inside stories of the cit2020-01-29’s most challenging health care crisis and most treasured medical triumphs.

Throughout the book, he thinks passionatel2020-01-29 about man2020-01-29 pressing issues facing Hong Kong, and wh2020-01-29 the2020-01-29 are not onl2020-01-29 challenges but also opportunities. 

In the end, this is a book more about the histor2020-01-29 and future of Hong Kong than the life of one of its inhabitants.
'''
for text in texts.split('.')[:-1:]:
    print(text)
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # # TF
    # model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
    # model.save_pretrained(MODEL)

    # text = "Good night 😊"
    # encoded_input = tokenizer(text, return_tensors='tf')
    # output = model(encoded_input)
    # scores = output[0][0].numpy()
    # scores = softmax(scores)

    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    for i in range(scores.shape[0]):
        l = labels[ranking[i]]
        s = scores[ranking[i]]
        print(f"{i+1}) {l} {np.round(float(s), 4)}")
