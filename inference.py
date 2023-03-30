import os.path
import re

import numpy as np
import soundfile
import torch

import utils
from models import SynthesizerTrn
from text import symbols, text_to_sequence
from text.zh_frontend import zh_to_pinyin, get_seg, get_sentence_positions, is_chinese_character

hps = utils.get_hparams_from_file("configs/hifigan.json")

net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model,
        config=hps.config)

pth = "/Volumes/Extend/下载/G_126500 (1).pth"
utils.load_checkpoint(pth, net_g)


zh_dict = [i.strip() for i in open("text/zh_dict2.dict").readlines()]
zh_dict = {i.split("\t")[0]: i.split("\t")[1] for i in zh_dict}


text = "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。人工智能可以对人的意识、思维的信息过程的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。"
# text = "下面给大家简单介绍一下怎么使用这个教程吧！首先我们要有魔法，才能访问到谷歌的云平台。点击连接并更改运行时类型，设置硬件加速器为GPU。然后，我们再从头到尾挨个点击每个代码块的运行标志。可能需要等待一定的时间。当我们进行到语音合成部分时，就可以更改要说的文本，并设置保存的文件名啦。"

segments = get_seg(text)
clean_txt = ''
for pair in segments:
        clean_txt += pair.word
seg_txt = "#1".join([p.word for p in segments])
clean_txt = clean_txt.replace("…", "...").replace("—", "-")

sp_position = get_sentence_positions(clean_txt)


pinyin = zh_to_pinyin(text)
phones = []
pros_phones = []
for idx, p in enumerate(pinyin.split(" ")):
        if p in zh_dict.keys():
                phones += zh_dict[p].split(" ")
                pros_phones += zh_dict[p].split(" ")
                pros_phones.append('^')
                if idx+1 in sp_position:
                        phones.append("$")
                        pros_phones.append("$")
encoding_txt =["[CLS]"]+ [ch for ch in clean_txt] +["[SEP]"]
bert_encodings = {line.strip():idx for idx, line in enumerate(open("tiny_bert/vocab.txt").readlines())}

try:
    encoding = [bert_encodings[i] for i in encoding_txt]
except:
    print(clean_txt)
sub = ''
txt2sub = []
idx = 0
for ch in encoding_txt:

    if is_chinese_character(ch):
        idx+=1
        sub += ch
        txt2sub.append(1)
        if idx in sp_position:
            sub += '.'
    else:
        txt2sub.append(0)

sub2sub = []
idx = 0
for ch in sub:
    if ch != ".":
        sub2sub.append(idx)
        idx += 1
    else:
        sub2sub.append(-1)

sub2phn = []
ph_count = 0
for ph in pros_phones:
    if ph =="^":
        sub2phn.append(ph_count)
        ph_count = 0
    elif ph =="$":
        assert ph_count==0
        sub2phn.append(1)
    else:
        ph_count += 1
if  len(sub2phn) != len(sub) or sum(sub2phn) != len(phones):
    print("skip", phones)
word2sub = []
idx = 0
for pair in segments:
    if bool(re.match(r'^[\u4e00-\u9fa5]+$', pair.word)):
        word2sub.append(len(pair.word))
        idx += len(pair.word)
        if idx in sp_position:
            word2sub.append(1)
if sum(word2sub) != len(sub):
    print("skip",phones)

outline = []
outline.append(" ".join(["{" + i.strip() + "}" for i in " ".join(phones).split('$')])) #0
outline.append(("{" + " ".join(pros_phones) + "}").replace("^", "1").replace("$ ", ""))#1
outline.append(" ".join(["1" for i in range(len(phones))]))#2
outline.append(seg_txt)#3
outline.append(clean_txt)#4
outline.append(" ".join([str(i) for i in txt2sub]))#5
outline.append(" ".join([str(i) for i in sub2phn]))#6
outline.append(" ".join([str(i) for i in encoding]))#7
outline.append(" ".join([i for i in encoding_txt]))#8
outline.append(" ".join([str(i) for i in word2sub]))#9
outline.append(" ".join(["1" for i in word2sub]))#10
outline.append(" ".join([str(i) for i in sub2sub]))#11
print("\n".join(outline))





x = torch.LongTensor(np.array(text_to_sequence(outline[0], []))).unsqueeze(0)
x_lengths = torch.LongTensor([x.shape[0]])
txt = torch.LongTensor(np.array(encoding)).unsqueeze(0)
txt2sub = torch.LongTensor(np.array(txt2sub)).unsqueeze(0)
sub2sub = torch.LongTensor(np.array(sub2sub)).unsqueeze(0)

sub2phn_m, sub2phn_e = utils.ali_mask([np.array(sub2phn)])
sub2phn_m =  torch.BoolTensor(sub2phn_m)
sub2phn_e = torch.LongTensor(sub2phn_e)
word2sub_m, word2sub_e = utils.ali_mask([np.array(word2sub)])
word2sub_m =  torch.BoolTensor(word2sub_m)
word2sub_e = torch.LongTensor(word2sub_e)
speakers = torch.LongTensor([0])

length_word = np.array(list())
for w2s in [np.array(word2sub)]:
    length_word = np.append(length_word, w2s.shape[0])
subword_len = np.array(list())
for s2s in [np.array(sub2sub)]:
    subword_len = np.append(subword_len, s2s.shape[0])

sent2word_m, sent2word_e = utils.ali_mask([[int(x)] for x in length_word])
sent2word_m =  torch.BoolTensor(sent2word_m)
sent2word_e = torch.LongTensor(sent2word_e)

sub_len = torch.LongTensor(subword_len)
length_word = torch.LongTensor(length_word)
# txt, txt_lengths = txt.cuda(rank, non_blocking=True), txt_len.cuda(rank, non_blocking=True)
# txt2sub = txt2sub.cuda(rank, non_blocking=True)
# sub2sub = sub2sub.cuda(rank, non_blocking=True)
# sub_len = sub_len.cuda(rank, non_blocking=True)
#
# speakers = speakers.cuda(rank, non_blocking=True)
#
#
# sub2phn_m = sub2phn_m.cuda(rank, non_blocking=True)
# word2sub_m = word2sub_m.cuda(rank, non_blocking=True)
# sent2word_m = sent2word_m.cuda(rank, non_blocking=True)
#
# sub2phn_e = sub2phn_e.cuda(rank, non_blocking=True)
# word2sub_e = word2sub_e.cuda(rank, non_blocking=True)
# sent2word_e = sent2word_e.cuda(rank, non_blocking=True)
with torch.no_grad():
    y_hat = net_g.infer(x, x_lengths, txt, None, txt2sub, speakers, None, None, sub2phn_m, sub2phn_e,
                                   word2sub_m, word2sub_e, sent2word_m, sent2word_e, None, sub2sub, sub_len)
print(y_hat.shape)
save_base = "samples/{}-{}.wav".format(text[:3], pth.split("/")[-1])
if os.path.exists(save_base):
    for i in range(10):
        if not os.path.exists(f"{save_base}.{i}.wav"):
            soundfile.write(f"{save_base}.{i}.wav", y_hat[0, 0, :].numpy(), 32000)
            break
else:
    soundfile.write(save_base, y_hat[0,0,:].numpy(),  32000)