import os

import librosa
import soundfile
from pydub import AudioSegment
from multiprocessing import Pool

# 定义输入和输出目录
from tqdm import tqdm

input_dir = '/Volumes/Extend/下载/vispeech/mfa_temp/wavs/zh/paimon'
output_dir = '/Volumes/Extend/下载/vispeech/mfa_temp/wavs/zh/paimon16k'

# 定义采样率为16K
sample_rate = 16000

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# 处理单个文件的函数
def process_file(filename):
    wav, sr = librosa.load(os.path.join(input_dir, filename), sr=sample_rate)
    soundfile.write(os.path.join(output_dir, filename), wav, sr)


if __name__ == '__main__':
    # 获取输入目录中所有.wav文件的文件名列表
    files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]

    # 设置进程池大小为4
    pool = Pool(processes=4)

    # 使用进度条显示进度
    for _ in tqdm(pool.imap_unordered(process_file, files), total=len(files)):
        pass

    # 关闭进程池
    pool.close()
    pool.join()
