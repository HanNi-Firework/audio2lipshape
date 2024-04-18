import copy

import numpy as np
import torch
import torch.nn as nn
from model.asr import ASR
from conformer import Conformer

from optim import Optimizer
import yaml
import argparse
from utils.audio_reader import DataReader, collate_fn
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.save_load import load_ckpt
from torchaudio import load, transforms
from utils.lips_visual import read_lip_class,write_video
import torchaudio
parser = argparse.ArgumentParser(description='Inference of model.')
parser.add_argument('--config', default="./config/conformer.yaml", type=str, help='Path of yaml')
paras = parser.parse_args()
config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)

dim = 80


def mix_if_necessary(waveform):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    waveform = waveform.squeeze()
    return waveform

def parse_audio(path):
    audio_conf = {"sample_rate": 16000, 'window_size': 0.04, 'window_stride': 0.00}
    n_fft = int(audio_conf['sample_rate'] * audio_conf["window_size"])
    win_length = n_fft
    # hop_length = int(self.audio_conf['sample_rate'] * self.audio_conf['window_stride'])
    hop_length = None
    waveform, sample_rate = load(path)
    waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=audio_conf['sample_rate'])
    print(waveform.shape)
    waveform = mix_if_necessary(waveform)
    print(waveform.shape)
    mel_spectrogram = transforms.MelSpectrogram(sample_rate=audio_conf['sample_rate'],
                                                n_fft=n_fft,
                                                win_length=win_length,
                                                hop_length=win_length,
                                                f_min=20.0,
                                                f_max=None,
                                                pad=0,
                                                n_mels=80,
                                                window_fn=torch.hamming_window,
                                                power=2,
                                                normalized=False,
                                                center=False,
                                                onesided=None,
                                                mel_scale='htk')(waveform)
    print(mel_spectrogram.shape)
    return torch.as_tensor(mel_spectrogram).transpose(0,1)


# def get_seg(series_daat, min_len=13, max_len=250):
#     seg = list()
#     temp = copy.deepcopy(series_daat)
#
#     if len(temp) != 0:
#         if len(temp) > max_len:
#             while len(temp) > max_len:
#                 seg_temp = temp[:max_len]
#                 seg.append(seg_temp)
#                 temp = temp[int(max_len):]   #with 20% overlap
#         if len(temp) >= min_len:
#             seg.append(temp)
#     return seg


# cuda = torch.cuda.is_available()
# device = torch.device('cuda' if cuda else 'cpu')
device = 'cpu'
# try:/
model = Conformer(num_classes=28, input_dim=dim, encoder_dim=512, num_encoder_layers=3).to(device)
# model = ASR(dim, 28, False, 0.0).to(device)
model_paras = list(model.parameters())
optimizer = Optimizer(model_paras, **config['hparas'])
model, _, _ = load_ckpt(model, optimizer, device, "./ckpts/comformer_3e4_9.pth", mode='eval')
model.eval()
audio_path = "C:\\Users\\NiH\\Downloads\\audiobox (2).wav"
mel_spectrogram = parse_audio(audio_path)
print(mel_spectrogram)
# data = get_seg(mel_spectrogram)
# print(len(data))
imgs, h, w = read_lip_class()
lips_idx = list()
with torch.no_grad():
    # for inputs in data:
    inputs = torch.as_tensor(mel_spectrogram)
    data_length = torch.as_tensor(inputs.shape[0])
    inputs = torch.unsqueeze(inputs, dim=0).cpu()
    # print(data_length.detach().numpy())
    data_length = torch.unsqueeze(data_length, dim=0).cpu()
    outputs, _ = model(inputs, data_length)
    outputs = outputs.detach().numpy()
    print(outputs.shape)
    lip_class = list()
    for item in outputs[0]:
        # print(item.shape)
        max_indices = np.argmin(item)
        max_elements = item[max_indices]
        lip_class.append(max_indices)
    lips_idx.extend(lip_class)
    # print(len(lip_class))
    print(lip_class)
    write_video(audio_path, lips_idx, "./lips_audio", imgs, h, w)