
from text.frontend.zh_frontend import Frontend
frontend = Frontend()


pu_symbols = ['!', '?', '…', ",", "."]

# print(_symbol_to_id)
pinyin2phones = {}

for line in open("text/zh_dict.dict").readlines():
    pinyin, phones = line.strip().split("\t")
    phones = phones.split(" ")
    pinyin2phones[pinyin] = phones

def pu_symbol_replace(data):
    chinaTab = ['！', '？', "…", "，", "。",'、', "..."]
    englishTab = ['!', '?', "…", ",", ".",",", "…"]
    for index in range(len(chinaTab)):
        if chinaTab[index] in data:
            data = data.replace(chinaTab[index], englishTab[index])
    return data

# def del_special_pu(data):
#     ret = ''
#     to_del = ["'", "\"", "“","", '‘', "’", "”"]
#     for i in data:
#         if i not in to_del:
#             ret+=i
#     return ret


def zh_to_phonemes(text):
    # 替换标点为英文标点
    text = pu_symbol_replace(text)
    phones = frontend.get_phonemes(text)[0]
    return phones

def get_seg(text):
    # 替换标点为英文标点
    text = pu_symbol_replace(text)
    seg = frontend.get_seg(text)
    return seg


def pinyin_to_phonemes(text):
    phones = []
    for pinyin in text.split(" "):
        try:
            phones += pinyin2phones[pinyin]
        except:
            print("词典中无此拼音：", pinyin)
    return phones

zh_dict = [i.strip() for i in open("text/zh_dict.dict").readlines()]
zh_dict = {i.split("\t")[0]: i.split("\t")[1] for i in zh_dict}

reversed_zh_dict = {}
all_zh_phones = set()
for k, v in zh_dict.items():
    reversed_zh_dict[v] = k
    [all_zh_phones.add(i) for i in v.split(" ")]

def phones_to_pinyins(phones):
    pinyins = ''
    accu_ph = []
    for ph in phones:
        accu_ph.append(ph)
        if ph not in all_zh_phones:
            # print(ph)
            assert len(accu_ph) == 1
            # pinyins += ph
            accu_ph = []
        elif " ".join(accu_ph) in reversed_zh_dict.keys():
            pinyins += " " + reversed_zh_dict[" ".join(accu_ph)]
            accu_ph = []
    assert  accu_ph==[]
    return pinyins.strip()

def zh_to_pinyin(text):
    phones = zh_to_phonemes(text)
    pinyin = phones_to_pinyins(phones)
    return pinyin

def is_chinese_character(ch):
    import re
    return bool(re.search(r'[\u4e00-\u9fa5]', ch))

def get_sentence_positions(clean_txt):
    positions = []

    res = []
    cum= ""
    for ch in clean_txt:
        cum += ch
        if not is_chinese_character(ch):
            if len(cum)>1:
                res.append(cum)
            cum=''
    current_pos=0
    sp_symbols = []
    for sentence in res[:-1]:
        positions.append(len(sentence)-1+current_pos)
        current_pos += len(sentence) -1
        # sp_symbols.append(sentence[-1])
    return positions