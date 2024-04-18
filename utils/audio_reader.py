import torchaudio

import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torchaudio import load
from torch.nn.utils.rnn import pad_sequence
import copy
import librosa


class DataReader(Dataset):
    def __init__(self, root_path, train=True):
        super(DataReader, self).__init__()
        self.root_path = root_path
        self.train = train
        self.files_path = list()
        self.get_data_path()
        self.audio_conf = {"sample_rate": 16000, 'window_size': 0.04, 'window_stride': 0.00}
        # self.data_validation()

    def __getitem__(self, index):
        audio_data, label_data = self.read_audio_label(index)
        return audio_data, label_data

    def __len__(self):
        return len(self.files_path)

    def data_validation(self):
        # temp = copy.deepcopy(self.files_path)
        # self.files_path = list()
        # size = len(temp)
        # for idx in range(size):
        #     audio_path = os.path.join(self.root_path, temp[idx]) + ".mp3"
        #     if idx %100 == 0:
        #         print(f"validation for {idx}/{size}")
        #     try:
        #         load(audio_path)
        #         self.files_path.append(temp[idx])
        #     except Exception as e:
        #         print(audio_path, e)
        print("data_validation")
        SR = 16000
        for path in self.files_path:
            audio_path = os.path.join(self.root_path, path) + ".mp3"
            audio_wav = librosa.core.load(audio_path, sr=SR)[0]
            audio_size = float(audio_wav.shape[-1]) / SR * 25
            label_path = os.path.join(self.root_path, path) + ".txt"
            label_data = self.parse_label(label_path)
            if int(audio_size) != len(label_data):
                print(path, audio_size, len(label_data))
        print("data_validation done")

    def get_data_path(self):
        prefix_list = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f"]
        # prefix_list = ["0"]
        for i in prefix_list:
            for j in prefix_list:
                prefix = str(i) + str(j)
                subfolder_status = os.path.join(self.root_path, prefix)
                files = os.listdir(subfolder_status)
                for file in files:
                    if file.endswith(".mp3"):
                        file_name = str(file).split(".")[0]
                        self.files_path.append(os.path.join(prefix, file_name))

    def read_audio_label(self, index):
        file_path = self.files_path[index]
        # print(self.files_path[index])

        # file_path = "d9\d9178dcaece3_1"
        audio_path = os.path.join(self.root_path, file_path) + ".mp3"
        label_path = os.path.join(self.root_path, file_path) + ".txt"
        audio_data = self.parse_audio(audio_path)
        label_data = self.parse_label(label_path)
        return torch.as_tensor(audio_data).transpose(0,1), torch.as_tensor(label_data)

    @staticmethod
    def parse_label(path):
        with open(path, "r", encoding='utf8') as f:
            for line in f:
                if len(line) != 0:
                    class_idx = line.split(" ")
        class_idx = [int(item)+1 for item in class_idx]
        return np.asarray(class_idx)

    def mix_if_necessary(self, waveform):
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.squeeze()
        return waveform

    def parse_audio(self, path):

        n_fft = int(self.audio_conf['sample_rate'] * self.audio_conf["window_size"])
        win_length = n_fft
        # hop_length = int(self.audio_conf['sample_rate'] * self.audio_conf['window_stride'])
        hop_length = None
        waveform, sample_rate = load(path)
        # waveform, _ = librosa.load(path, sr=self.audio_conf["sample_rate"])
        # print(waveform.shape)
        waveform = self.mix_if_necessary(waveform)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=self.audio_conf['sample_rate'],
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

        return mel_spectrogram


# def collate_fn(data):
#     data_length_x = [sq[0].shape[0] for sq in data]
#     data_length_y = [sq[1].shape[0] for sq in data]
#     x = [i[0] for i in data]
#     # print(x[0].shape)
#     y = [i[1] for i in data]
#     # print(x)
#     x = pad_sequence(x, batch_first=True, padding_value=0.0)
#     y = pad_sequence(y, batch_first=True, padding_value=0)
#     # print(x, data_length_x, y, data_length_y)
#     print(len(x), len(data_length_x), len(y), len(data_length_y))
#     print(len(x[0]), len(data_length_x), len(y[0]), len(data_length_y))
#
#     return x, data_length_x, y, data_length_y


def collate_fn(data):
    data_length_x = [sq[0].shape[0] for sq in data]
    data_length_y = [sq[1].shape[0] for sq in data]
    x = [i[0] for i in data]
    # print(x[0].shape)
    y = [i[1] for i in data]
    # print(x)
    x = pad_sequence(x, batch_first=True, padding_value=0.0)
    y = pad_sequence(y, batch_first=True, padding_value=0)
    y = torch.nn.functional.one_hot(torch.from_numpy(np.asarray(y, dtype=np.int64)), 28)

    # print(x, data_length_x, y, data_length_y)
    # print(len(x), len(data_length_x), len(y), len(data_length_y))
    # print(len(x[0]), len(data_length_x), len(y[0]), len(data_length_y))

    return x, data_length_x, y, data_length_y


# train_dataset = DataReader(root_path="E:\\DH\\audio_lips")
# train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
# iterator = iter(train_dataloader)
# audio, data_length, label = next(iterator)
# print(audio.shape, data_length)
