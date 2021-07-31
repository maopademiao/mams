# -*- coding: utf-8 -*-
# @Time    : 2020-05-08 10:50
# @Author  : Zheng Lei
# @Site    : 
# @File    : preprocess.py
# @Description: 添加注释

from xml.etree.ElementTree import parse
import codecs
import json

def get_polarity_words(path):
    polarity_words = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            if line != '':
                polarity_words.append(line)
    return polarity_words


def get_term_polarity(path):
    neg_words = get_polarity_words('negative-words.txt')
    pos_words = get_polarity_words('positive-words.txt')

    tree = parse(path)
    sentences = tree.getroot()
    neg, pos = set(), set()
    for i, sentence in enumerate(sentences):
        print(i)
        text = sentence.find('text')
        if text is None:
            continue
        text = str(text.text).lower().strip()
        for w in text.split():
            if w in pos_words:
                pos.add(w)
            if w in neg_words:
                neg.add(w)
    return pos, neg


def parse_sentence_term_split(path, lowercase=False):

    tree = parse(path)
    sentences = tree.getroot()
    datas = []
    split_char = '__split__'
    for i, sentence in enumerate(sentences):
        print(i)
        text = sentence.find('text')
        if text is None:
            continue
        text = str(text.text)
        if lowercase:
            text = text.lower()
        ts = text.split(',')

        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue

        for aspectTerm in aspectTerms:
            term = aspectTerm.get('term')
            if lowercase:
                term = term.lower()
            polarity = aspectTerm.get('polarity')
            start = int(aspectTerm.get('from'))
            end = int(aspectTerm.get('to'))
            assert text[start:end] == term
            lens = len(datas)
            if len(ts) == 1 or term.find(',') > -1:
                piece = ts[0] + split_char + term + split_char + polarity + split_char + str(start) + split_char + str(end)
                datas.append(piece)
            else:
                for t in ts:
                    if t.find(term) > -1:
                        text1 = t
                        start = t.index(term)
                        end = start + len(term)
                        assert text1[start:end] == term
                        piece = text1 + split_char + term + split_char + polarity + split_char + str(start) + split_char + str(end)
                        datas.append(piece)
                        break
            assert lens + 1 == len(datas)
    return datas


def get_dev_result():
    d = {
        0: 'positive',
        1: 'negative',
        2: 'neutral'
    }
    res = json.load(open('result.json'))
    preds, reals = res['pred'], res['real']

    error_res_p_n = []
    error_res_p_t = []
    error_res_n_p = []
    error_res_n_t = []
    error_res_t_p = []
    error_res_t_n = []

    tree = parse('dev.xml')
    sentences = tree.getroot()
    index = 0
    for i, sentence in enumerate(sentences):
        print(i)
        text = sentence.find('text')
        text = text.text
        aspectTerms = sentence.find('aspectTerms')
        if aspectTerms is None:
            continue
        for aspectTerm in aspectTerms:
            term = aspectTerm.get('term')
            polarity = aspectTerm.get('polarity')
            assert d[reals[index]] == polarity
            if d[preds[index]] != polarity:
                r = {
                    'text': text,
                    'term': term,
                    'real': polarity,
                    'pred': d[preds[index]],
                }
                if polarity == 'positive':
                    if d[preds[index]] == 'negative':
                        error_res_p_n.append(r)
                    elif d[preds[index]] == 'neutral':
                        error_res_p_t.append(r)
                elif polarity == 'negative':
                    if d[preds[index]] == 'positive':
                        error_res_n_p.append(r)
                    elif d[preds[index]] == 'neutral':
                        error_res_n_t.append(r)
                elif polarity == 'neutral':
                    if d[preds[index]] == 'positive':
                        error_res_t_p.append(r)
                    elif d[preds[index]] == 'negative':
                        error_res_t_n.append(r)
            index += 1
    print('positive - negative:', len(error_res_p_n))
    with codecs.open('pos_neg.json', 'w', encoding='utf-8') as f:
        json.dump(error_res_p_n, f, indent=4, ensure_ascii=False)

    print('positive - neutral:', len(error_res_p_t))
    with codecs.open('pos_neut.json', 'w', encoding='utf-8') as f:
        json.dump(error_res_p_t, f, indent=4, ensure_ascii=False)

    print('negative - positive:', len(error_res_n_p))
    with codecs.open('neg_pos.json', 'w', encoding='utf-8') as f:
        json.dump(error_res_n_p, f, indent=4, ensure_ascii=False)

    print('negative - neutral:', len(error_res_n_t))
    with codecs.open('neg_neut.json', 'w', encoding='utf-8') as f:
        json.dump(error_res_n_t, f, indent=4, ensure_ascii=False)

    print('neutral - positive:', len(error_res_t_p))
    with codecs.open('neut_pos.json', 'w', encoding='utf-8') as f:
        json.dump(error_res_t_p, f, indent=4, ensure_ascii=False)

    print('neutral - negative:', len(error_res_t_n))
    with codecs.open('neut_neg.json', 'w', encoding='utf-8') as f:
        json.dump(error_res_t_n, f, indent=4, ensure_ascii=False)

    # positive - negative: 47
    # 出现否定词
    # "Despite him not being my designated server"
    # "But wait, it gets even better, the mussels were so fishy"
    # "service aren't nearly as nice as the decor"
    # "and the prices are low."
    # 标错
    # <sentence>
    # 		<text>Decor is old, but the bathrooms are clean and updated.</text>
    # 		<aspectTerms>
    # 			<aspectTerm from="0" polarity="positive" term="Decor" to="5"/>
    # 			<aspectTerm from="22" polarity="negative" term="bathrooms" to="31"/>
    # 		</aspectTerms>
    # 	</sentence>

    # positive - neutral: 84
    # 出现较多的aa和bb
    # 标错
    # <text>For $65, you will get a 5-6 course tasting menu encompassing a variety of raw fish, cooked fish, pastas, dessert, etc.</text>
    # 		<aspectTerms>
    # 			<aspectTerm from="43" polarity="positive" term="menu" to="47"/>
    # 			<aspectTerm from="74" polarity="positive" term="raw fish" to="82"/>
    # 			<aspectTerm from="97" polarity="neutral" term="pastas" to="103"/>
    # 			<aspectTerm from="105" polarity="neutral" term="dessert" to="112"/>
    # 		</aspectTerms>
    # <text>Other guests enjoyed pizza, santa fe chopped salad and fish and chips.</text>
    # 		<aspectTerms>
    # 			<aspectTerm from="21" polarity="positive" term="pizza" to="26"/>
    # 			<aspectTerm from="28" polarity="positive" term="santa fe chopped salad" to="50"/>
    # 			<aspectTerm from="55" polarity="neutral" term="fish and chips" to="69"/>
    # 		</aspectTerms>
    # <text>Please have the SPINACH DIP, CAESAR SALAD, HAWAIIAN RIB-EYE and the BROWNIE for dessert.</text>
    # 		<aspectTerms>
    # 			<aspectTerm from="16" polarity="positive" term="SPINACH DIP" to="27"/>
    # 			<aspectTerm from="29" polarity="positive" term="CAESAR SALAD" to="41"/>
    # 			<aspectTerm from="80" polarity="neutral" term="dessert" to="87"/>
    # 		</aspectTerms>
    # 识别错误
    # my favorite is

    # negative - positive: 44
    # 标错
    # <text>The servers, casual in their striped button-downs, anticipate and fulfill needs as if they were trained as mind readers.</text>
    # 		<aspectTerms>
    # 			<aspectTerm from="4" polarity="negative" term="servers" to="11"/>
    # 			<aspectTerm from="29" polarity="neutral" term="striped button-downs" to="49"/>
    # 		</aspectTerms>
    # 区分不出价格
    # "text": "We had a nice bottle of wine, dinner, and dessert for under $200.",
    #         "term": "dessert",
    #         "real": "positive",
    #         "pred": "neutral"
    # 区分不出是否标错？好的形容词后面加了though
    # {
    #         "text": "The service was adequate though we did need to keep asking for water and the drinks from the bar took a long time- plus the restaraurant was not crowded.",
    #         "term": "service",
    #         "real": "negative",
    #         "pred": "positive"
    #     },

    # negative - neutral: 56
    # 没有明显倾向的词，要放在语境里才能看出来
    # the food is not nearly good enough to wait an hour before seeing your server
    # While the deocr in the dining room is less than inviting
    # 非实体类型的词
    # the obligatory grousing waiters - obligatory - 强制性的
    # After waiting for 90 minutes - waiting
    # 形容词识别不出来情感倾向
    # The food seems so-so

    # neutral - positive: 90
    # 上下文存在表达情感积极的词
    # Appetizers and entrees were merely adequate, but you can't beat the pool room for atmosphere - Appetizers
    # But wait, it gets even better, the mussels were so fishy - wait
    # if not for the drinks, then definitely for the unique approach to mexican. - mexican
    # there's an extensive veggy menu, as well as many meat dishes - meat

    # neutral - negative: 83
    # 同上，上下文存在表达情感消极的词
    # On the other hand, the soup was so clear and you taste no salt - salt
    # The small bar is always packed with people - bar
    # and the friend shrimp had too much curry. - shrimp 朋友虾吃了太多咖喱。
    # improve the decor and dine-in service, but DON'T change the portions or the recipes. - portions
    # it was impossible for guests and waiter to move without bumping you. - waiter



if __name__ == '__main__':
    # pos1, neg1 = get_term_polarity('train.xml')
    # pos2, neg2 = get_term_polarity('dev.xml')
    # pos3, neg3 = get_term_polarity('dev.xml')
    # pos = pos1.union(pos2).union(pos3)
    # neg = neg1.union(neg2).union(neg3)
    # sentiments = {
    #     "positive": list(pos),
    #     "negative": list(neg),
    #     "neutral": ["neutral"]
    # }
    # with codecs.open('sentiment_dict.json', 'w', encoding='utf-8') as f:
    #     json.dump(sentiments, f, indent=4, ensure_ascii=False)
    get_dev_result()
