import numpy as np
import torch
import torch.nn as nn
from conformer import Conformer
from model.asr import ASR
from optim import Optimizer
import yaml
import argparse
from utils.audio_reader import DataReader, collate_fn
from torch.utils.data import DataLoader
from utils.timer import Timer
from utils.save_load import save_checkpoint


parser = argparse.ArgumentParser(description='Training conformer.')
parser.add_argument('--config', default="./config/conformer.yaml", type=str, help='Path of yaml')
parser.add_argument('--GRAD_CLIP', default=5.0, type=float)
parser.add_argument('--epoch', default=30, type=int)
paras = parser.parse_args()
config = yaml.load(open(paras.config, 'r'), Loader=yaml.FullLoader)

dim = 80

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
# try:
criterion = nn.CrossEntropyLoss().to(device)
model = Conformer(num_classes=28, input_dim=dim, encoder_dim=512, num_encoder_layers=3).to(device)
# model = ASR(dim, 28, False, 0.0).to(device)
model_paras = list(model.parameters())
# print(model_paras)
optimizer = Optimizer(model_paras, **config['hparas'])
# Forward propagate

# inputs = torch.rand(batch_size, sequence_length, dim).to(device)
# input_lengths = torch.LongTensor([12345, 12300, 12000])
# targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
#                             [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
#                             [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
# target_lengths = torch.LongTensor([9, 8, 7])
# outputs, output_lengths = model(inputs, input_lengths)

# optimizer = Optimizer(model.model_paras)
batch_size = 24
train_dataset = DataReader(root_path="E:\\DH\\audio_lips_with_max_fixed")
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
global_step = 0
total_loss = 0.0
from torch import autograd

# with autograd.detect_anomaly():
timer = Timer()
for epoch in range(paras.epoch):
    iterator = iter(train_dataloader)
    for step, (audio, data_length_audio, label, data_length_label) in enumerate(iterator):
        inputs = audio.to(device)
        targets = label.to(device)
        # targets = torch.nn.functional.one_hot(torch.Tensor.int(targets), 28)
        data_length_audio = torch.LongTensor(data_length_audio)
        data_length_label = torch.LongTensor(data_length_label)

        optimizer.pre_step(global_step)
        # print(inputs.shape, targets.shape, data_length_audio, data_length_label)
        # print(inputs.shape, targets.shape)
        outputs, output_lengths = model(inputs, data_length_audio)
        # print(outputs.shape, targets.shape)
        # Calculate CTC Loss
        # print(outputs.shape, targets.shape)
        outputs = outputs.transpose(0, 1).flatten(0, 1)
        targets = targets.flatten(0, 1)
        # print(outputs.shape, targets.shape)
        loss = criterion(outputs, targets.float())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), paras.GRAD_CLIP)
        optimizer.step()
        total_loss += loss.item()
        if (global_step+1) % 100 == 0:
            timer(f"epoch:{epoch}, step:{step}, lr:{optimizer.get_opt_state_dict()['param_groups'][0]['lr']}, loss:{total_loss/100}")
            total_loss = 0.0
        global_step += 1
    f_name = f"comformer_5e5_{epoch}.pth"
    save_checkpoint(model, optimizer, global_step, "./ckpts",  f_name, loss)

# except Exception as e:
#     print(e)