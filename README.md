# HierTTS
HierTTS: Expressive End-to-End Text-to-Waveform using Multi-Scale Hierarchical Variational Auto-encoder

实验失败，多说人有点问题，音质比hifigan差，多层kl loss无法正常收敛，韵律并不理想，且时长预测几乎没有多样性（应该是没有正确收敛导致）
### Zengqiang Shang, Pengyang Shi, Pengyuan Zhang

In our recent [paper](https://www.mdpi.com/2076-3417/13/2/868), we propose HierTTS: Expressive End-to-End Text-to-Waveform using Multi-Scale Hierarchical Variational Auto-encoder

<table style="width:100%">
  <tr>
    <th>HierTTS</th>
  </tr>
  <tr>
    <td><img src="resources/zeng1-1.png" alt="HierTTS" height="450"></td>
  </tr>
</table>



## Training Exmaple
bash run.sh

## Inference Example
bash infer.sh