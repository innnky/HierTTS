import os
import re

import librosa
import numpy as np
import parselmouth
import tqdm
from scipy.interpolate import interp1d
import tgt

from preprocessors.utils import get_alignment_word_boundary

from text.symbols import symbols
from text.zh_frontend import get_seg
# print(symbols)
preprocessed_out_path = "norm_preprocessed_hifi16_bert/paimon"
preprocessed_out_path = "/Volumes/Extend/下载/paimon_hier"
tgt_base_path = "/Volumes/Extend/下载/vispeech/mfa_temp/textgrids/zh/paimon"
wav_base_path = "/Volumes/Extend/下载/vispeech/mfa_temp/wavs/zh/paimon16k"
transcription_path = "/Volumes/Extend/下载/vispeech/data/zh/paimon/transcription_raw.txt"
sampling_rate = 16000
hop = 200
nfft = 1024
win_length = 800

def stft(y):
    return librosa.stft(
        y=y,
        n_fft=nfft,
        hop_length=hop,
        win_length=win_length,
    )

def rawenergy(y):
    # Extract energy
    S = librosa.magphase(stft(y))[0]
    e = np.sqrt(np.sum(S ** 2, axis=0))  # np.linalg.norm(S, axis=0)
    return e.squeeze()  # (Number of frames) => (654,)

def get_energy(wav, p_len=None):
    e = rawenergy(wav)
    if p_len is None:
        p_len = wav.shape[0] // 512
    assert e.shape[0] -p_len <2 ,(e.shape[0] ,p_len)
    e = e[: p_len]
    return e


def get_pitch(wav_data,lll):
    """
    :param wav_data: [T]
    :param mel: [T, 80]
    :param config:
    :return:
    """
    hop_length = hop

    time_step = hop_length / sampling_rate * 1000
    f0_min = 80
    f0_max = 750

    f0 = parselmouth.Sound(wav_data, sampling_rate).to_pitch_ac(
        time_step=time_step / 1000, voicing_threshold=0.6,
        pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array["frequency"]
    lpad = 2
    rpad = lll - len(f0) - lpad
    assert 0<=rpad<=2,(len(f0), lll, len(wav_data)//hop_length)
    assert 0<=( lll- len(wav_data)//hop_length)<=1
    f0 = np.pad(f0, [[lpad, rpad]], mode="constant")

    return f0

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
    for tgt_path in tqdm.tqdm(os.listdir(tgt_base_path)):
        if tgt_path.endswith("TextGrid"):
            name = tgt_path.replace(".TextGrid", "")
            if os.path.exists(f"{wav_base_path}/{name}.wav"):
                textgrid = tgt.io.read_textgrid(f"{tgt_base_path}/{tgt_path}")
                phone_tier = textgrid.get_tier_by_name("phones")
                word_tier = textgrid.get_tier_by_name("words")
                res = get_alignment_word_boundary(phone_tier, word_tier, sampling_rate, hop, "Chinese", return_tail=True)
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

                audio, sr = librosa.load(f"{wav_base_path}/{name}.wav", sr=sampling_rate)
                audio = audio[int(start_time*sampling_rate):int(end_time*sampling_rate)]
                pitch = get_pitch(audio, sum(durations))
                nonzero_ids = np.where(pitch != 0)[0]
                try:
                    interp_fn = interp1d(
                        nonzero_ids,
                        pitch[nonzero_ids],
                        fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                        bounds_error=False,
                    )
                    pitch = interp_fn(np.arange(0, len(pitch)))
                except:
                    pass
                pos = 0
                for i, d in enumerate(durations):
                    if d > 0:
                        pitch[i] = np.mean(pitch[pos: pos + d])
                    else:
                        pitch[i] = 0
                    pos += d
                pitch = pitch[: len(durations)]

                energy = get_energy(audio, sum(durations))
                pos = 0
                for i, d in enumerate(durations):
                    if d > 0:
                        energy[i] = np.mean(energy[pos: pos + d])
                    else:
                        energy[i] = 0
                    pos += d
                energy = energy[: len(durations)]
                durations = np.array(durations)
                np.save(f"{preprocessed_out_path}/alignment/multispeaker-ali-{name}.npy", durations)
                np.save(f"{preprocessed_out_path}/audio/multispeaker-audio-{name}.npy", audio)
                np.save(f"{preprocessed_out_path}/energy/multispeaker-energy-{name}.npy", energy)
                np.save(f"{preprocessed_out_path}/f0/multispeaker-f0-{name}.npy", pitch)

                # break





