import os
import re

import tgt

from preprocessors.utils import get_alignment_word_boundary

from text.symbols import symbols
from text.zh_frontend import get_seg
# print(symbols)
preprocessed_out_path = "norm_preprocessed_hifi16_bert/paimon"
tgt_base_path = "/Volumes/Extend/下载/vispeech/mfa_temp/textgrids/zh/paimon"
wav_base_path = "/Volumes/Extend/下载/vispeech/mfa_temp/wavs/zh/paimon"
transcription_path = "/Volumes/Extend/下载/vispeech/data/zh/paimon/transcription_raw.txt"

os.makedirs(preprocessed_out_path, exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/alignment", exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/audio", exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/energy", exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/f0", exist_ok=True)
os.makedirs(f"{preprocessed_out_path}/spec", exist_ok=True)

lab_dict ={}
for line in open(transcription_path).readlines():
    name, _, txt =line.strip().split("|")
    lab_dict[name] = txt

bert_encodings = {line.strip():idx for idx, line in enumerate(open("tiny_bert/vocab.txt").readlines())}
with open(f"{preprocessed_out_path}/train.txt", "w") as f:
    for tgt_path in os.listdir(tgt_base_path):
        if tgt_path.endswith("TextGrid"):
            name = tgt_path.replace(".TextGrid", "")
            if os.path.exists(f"{wav_base_path}/{name}.wav"):
                textgrid = tgt.io.read_textgrid(f"{tgt_base_path}/{tgt_path}")
                phone_tier = textgrid.get_tier_by_name("phones")
                word_tier = textgrid.get_tier_by_name("words")
                res = get_alignment_word_boundary(phone_tier, word_tier, 16000, 200, "Chinese", return_tail=True)
                phones, durations, start_time, end_time, pros_phones, mask = res
                txt = lab_dict[name]
                if re.search(r'[a-zA-Z0-9]', txt):
                    print("跳过",txt)
                    continue
                segments = get_seg(txt)
                clean_txt = ''
                for pair in segments:
                    clean_txt+=pair.word
                seg_txt = "#1".join([p.word for p in segments])
                clean_txt = clean_txt.replace("…","...").replace("—","-")
                encoding_txt =["[CLS]"]+ [ch for ch in clean_txt] +["[SEP]"]
                try:
                    encoding = [bert_encodings[i] for i in encoding_txt]
                except:
                    print(clean_txt)
                    continue
                sub = ''
                txt2sub = []
                for ch in encoding_txt:
                    if re.search(r'[\u4e00-\u9fa5]', ch):
                        sub += ch
                        txt2sub.append(1)
                    else:
                        txt2sub.append(0)
                sub2sub = [i for i in range(len(sub))]

                sub2phn = []
                ph_count = 0
                for ph in pros_phones:
                    if ph =="^":
                        sub2phn.append(ph_count)
                        ph_count = 0
                    else:
                        ph_count += 1
                if  len(sub2phn) != len(sub) or sum(sub2phn) != len(phones):
                    print("skip",txt, phones)
                    continue

                word2sub = []
                for pair in segments:
                    if bool(re.match(r'^[\u4e00-\u9fa5]+$', pair.word)):
                        word2sub.append(len(pair.word))
                if sum(word2sub) != len(sub):
                    print("skip",txt,phones)
                    continue

                outline = []
                outline.append(name)
                outline.append("{"+" ".join(phones)+"}")
                outline.append(("{"+" ".join(pros_phones)+"}").replace("^", "1"))
                outline.append(" ".join(["1" for i in range(len(phones))]))
                outline.append(seg_txt)
                outline.append(clean_txt)
                outline.append(" ".join([str(i) for i in txt2sub]))
                outline.append(" ".join([str(i) for i in sub2phn]))
                outline.append(" ".join([str(i) for i in encoding]))
                outline.append(" ".join([i for i in encoding_txt]))
                outline.append(" ".join([str(i) for i in word2sub]))
                outline.append(" ".join(["1" for i in word2sub]))
                outline.append(" ".join([str(i) for i in sub2sub]))
                outline.append("\n")
                f.write("|".join(outline))
                # print(clean_txt)
                # for pair in segments:
                #     print(pair.word)

                # print(res)
                # break





