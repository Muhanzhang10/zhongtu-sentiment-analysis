import pandas as pd
from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
import csv
import urllib.request


def run():
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

    path = r'data/海外.xlsx'  #输入文件路径
    input = pd.read_excel(path)
    out = input.loc[input['语种'] == 'eng']
    china = ['china', 'chinese',  'mandarin', 'sino', 'zhongguo', 'peking', 'guangzhou', 'shenzhen']
    us = ['USA', 'US', 'america', 'the united states']
    provinces = {
    "上海": "Shanghai",
    "云南": "Yunnan",
    "内蒙古": "Inner Mongolia",
    "北京": "Beijing",
    "台湾": "Taiwan",
    "吉林": "Jilin",
    "四川": "Sichuan",
    "天津": "Tianjin",
    "宁夏": "Ningxia",
    "安徽": "Anhui",
    "山东": "Shandong",
    "山西": "Shanxi",
    "广东": "Guangdong",
    "广西": "Guangxi",
    "新疆": "Xinjiang",
    "江苏": "Jiangsu",
    "江西": "Jiangxi",
    "河北": "Hebei",
    "河南": "Henan",
    "浙江": "Zhejiang",
    "海南": "Hainan",
    "湖北": "Hubei",
    "湖南": "Hunan",
    "澳门": "Macao",
    "甘肃": "Gansu",
    "福建": "Fujian",
    "西藏": "Tibet",
    "贵州": "Guizhou",
    "辽宁": "Liaoning",
    "重庆": "Chongqing",
    "陕西": "Shaanxi",
    "青海": "Qinhai",
    "香港": "Hong Kong",
    "黑龙江": "Heilongjiang"
    }
    for key in provinces:
        china.append(provinces[key].lower())

    states = ["Alaska", "Alabama", "Arkansas", "American Samoa", "Arizona", "California", "Colorado", "Connecticut", "District ", "of Columbia", "Delaware", "Florida", "Georgia", "Guam", "Hawaii", "Iowa", "Idaho", "Illinois", "Indiana", "Kansas", "Kentucky", "Louisiana", "Massachusetts", "Maryland", "Maine", "Michigan", "Minnesota", "Missouri", "Mississippi", "Montana", "North Carolina", "North Dakota", "Nebraska", "New Hampshire", "New Jersey", "New Mexico", "Nevada", "New York", "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Puerto Rico", "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Virginia", "Virgin Islands", "Vermont", "Washington", "Wisconsin", "West Virginia", "Wyoming"]
    for state in states:
        us.append(state)

    topic = []
    positive = []
    neutral = []
    negative = []
    negative_sentences = []

    for index, row in out.iterrows():
        title = row['题名']
        intro = row['简介']
        keywords = str(row['关键词'])
        category = str(row['中图分类法']).lower()
        texts = str(title) + '. ' +  str(intro) 
        t = ''
        relevant = False
        for c in china:
            if c in texts.lower() or c in category.lower() or c in keywords.lower():
                t += 'China '
                relevant = True
                break
        for u in us:
            if u in texts or u in category or u in keywords:
                t += 'US'
                relevant = True
                break
        if relevant == False:
            out = out.drop([index])
            continue
        topic.append(t)
        
        pscore = 0 #positive score
        nuscore = 0 #neutral score
        nscore = 0 #negative score
        count = 0
        negative_sentence = []
        for text in (str(intro)).split('.'):
            #print(index)
            #print(text)
            if len(text) > 514:
                continue
            if text == '':
                continue
            count += 1
            text = preprocess(text)
            encoded_input = tokenizer(text, return_tensors='pt')
            output = model(**encoded_input)
            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

            ranking = np.argsort(scores)
            ranking = ranking[::-1]
            for i in range(scores.shape[0]):
                l = labels[ranking[i]]
                s = scores[ranking[i]]
                if l == 'positive':
                    pscore += s
                if l == 'neutral':
                    nuscore += s
                if l == 'negative':
                    if s >= 0.5:
                        print(text)
                        negative_sentence.append(text)
                    nscore += s
                #print(f"{i+1}) {l} {np.round(float(s), 4)}")

        if count == 0:
            positive.append(0)
            neutral.append(0)
            negative.append(0)
            negative_sentences.append(negative_sentence)
        else:
            positive.append(pscore / count)
            neutral.append(nuscore / count)
            negative.append(nscore / count)
            negative_sentences.append(negative_sentence)
        print('----------------------------------------------------------')


    out['positive'] = positive
    out['neutral'] = neutral
    out['negative'] = negative
    out['topic'] = topic
    out['negative_sentences'] = negative_sentences

    out = out[['中图ID', '题名', '作者（编者）', '语种', '简介', '中图分类法', '关键词', 'positive', 'neutral', 'negative', 'topic', 'negative_sentences']]

    # write DataFrame to excel
    out.to_excel(r'output/海外sentiment.xlsx', index = False) #输入存储文件路径
    print('DataFrame is written to Excel File successfully.')
    return 


run()



