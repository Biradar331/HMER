"""
CPU-only training script for Handwritten Mathematical Expression Recognition.

Notes:
- Replaces GPU/DataParallel calls with CPU device usage.
- Loads pretrained densenet weights with map_location=device if available.
- Ensure required data files (offline-train.pkl, train_caption.txt, offline-test.pkl, test_caption.txt)
  and dictionary.txt are present at project root.
- Training on CPU can be extremely slow; reduce epochs / batch_size for testing.
"""
import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from data_iterator import dataIterator
from Densenet_torchvision import densenet121
from Attention_RNN import AttnDecoderRNN
from PIL import Image

# Device: force CPU
device = torch.device("cpu")

# compute Levenshtein / WER
def cmp_result(label, rec):
    dist_mat = np.zeros((len(label) + 1, len(rec) + 1), dtype='int32')
    dist_mat[0, :] = np.arange(len(rec) + 1)
    dist_mat[:, 0] = np.arange(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i, j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            dist_mat[i, j] = min(hit_score, ins_score, del_score)
    dist = dist_mat[len(label), len(rec)]
    return dist, len(label)

def load_dict(dictFile):
    with open(dictFile, 'r', encoding='utf-8') as fp:
        stuff = fp.readlines()
    lexicon = {}
    for l in stuff:
        w = l.strip().split()
        if not w:
            continue
        lexicon[w[0]] = int(w[1])
    print('total words/phones', len(lexicon))
    return lexicon

# Paths and hyperparams (adjust for CPU testing)
datasets = ['./offline-train.pkl', './train_caption.txt']
valid_datasets = ['./offline-test.pkl', './test_caption.txt']
dictionaries = ['./dictionary.txt']
batch_Imagesize = 500000
valid_batch_Imagesize = 500000
batch_size = 1           # reduce for CPU quick tests
batch_size_t = 1
maxlen = 48
maxImagesize = 100000
hidden_size = 256
teacher_forcing_ratio = 1.0
lr_rate = 0.01
flag = 0
exprate = 0.0

# Load dictionary and reverse map
worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * (max(worddicts.values()) + 1)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk

# Load precomputed pkl batches
train, train_label = dataIterator(
    datasets[0], datasets[1], worddicts, batch_size=1,
    batch_Imagesize=batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize
)
len_train = len(train)

test, test_label = dataIterator(
    valid_datasets[0], valid_datasets[1], worddicts, batch_size=1,
    batch_Imagesize=valid_batch_Imagesize, maxlen=maxlen, maxImagesize=maxImagesize
)
len_test = len(test)

# Dataset wrapper
class custom_dset(data.Dataset):
    def __init__(self, train, train_label):
        self.train = train
        self.train_label = train_label

    def __getitem__(self, index):
        train_setting = torch.from_numpy(np.array(self.train[index])).float()
        label_setting = torch.from_numpy(np.array(self.train_label[index])).long()
        size = train_setting.size()
        # reshape to (1, H, W) per sample entry
        train_setting = train_setting.view(1, size[-2], size[-1])
        label_setting = label_setting.view(-1)
        return train_setting, label_setting

    def __len__(self):
        return len(self.train)

off_image_train = custom_dset(train, train_label)
off_image_test = custom_dset(test, test_label)

# Collate: pad images and labels to uniform size in batch
def collate_fn(batch):
    batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
    imgs, labels = zip(*batch)

    # compute max H,W
    max_h = 0
    max_w = 0
    for it in imgs:
        size = it.size()
        if size[1] > max_h:
            max_h = size[1]
        if size[2] > max_w:
            max_w = size[2]

    img_padding_mask = None
    for ii in imgs:
        ii = ii.float()
        img_h = ii.size(1)
        img_w = ii.size(2)
        mask = torch.ones(1, img_h, img_w, dtype=torch.float32) * 255.0
        img_mask_sub = torch.cat((ii, mask), dim=0)
        pad_h = max_h - img_h
        pad_w = max_w - img_w
        m = torch.nn.ZeroPad2d((0, pad_w, 0, pad_h))
        padded = m(img_mask_sub).unsqueeze(0)
        if img_padding_mask is None:
            img_padding_mask = padded
        else:
            img_padding_mask = torch.cat((img_padding_mask, padded), dim=0)

    # labels padding
    max_len = len(labels[0]) + 1
    label_padding = None
    for ii in labels:
        ii = ii.long().unsqueeze(0)
        ii_len = ii.size(1)
        m = torch.nn.ZeroPad2d((0, max_len - ii_len, 0, 0))
        ii_pad = m(ii)
        if label_padding is None:
            label_padding = ii_pad
        else:
            label_padding = torch.cat((label_padding, ii_pad), dim=0)

    img_padding_mask = img_padding_mask / 255.0
    return img_padding_mask, label_padding

# DataLoaders (use num_workers=0 on Windows / CPU)
train_loader = torch.utils.data.DataLoader(
    dataset=off_image_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)
test_loader = torch.utils.data.DataLoader(
    dataset=off_image_test,
    batch_size=batch_size_t,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=0,
)

# Training routine for a single mini-batch
def my_train(target_length, attn_decoder1,
             output_highfeature, output_area, y, criterion, encoder_optimizer1, decoder_optimizer1,
             x_mean, dense_input, h_mask, w_mask, decoder_input, decoder_hidden, attention_sum, decoder_attention):
    loss = 0.0
    use_teacher_forcing = (random.random() < teacher_forcing_ratio)
    flag_z = [0] * batch_size

    encoder_optimizer1.zero_grad()
    decoder_optimizer1.zero_grad()

    if use_teacher_forcing:
        y_work = y
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention, attention_sum = attn_decoder1(
                decoder_input, decoder_hidden, output_highfeature, output_area, attention_sum,
                decoder_attention, dense_input, batch_size, h_mask, w_mask, []
            )
            y_work = y_work.unsqueeze(0)
            for i in range(batch_size):
                if int(y_work[0][i][di]) == 0:
                    flag_z[i] += 1
                    if flag_z[i] > 1:
                        continue
                    else:
                        loss += criterion(decoder_output[i], y_work[:, i, di])
                else:
                    loss += criterion(decoder_output[i], y_work[:, i, di])
            if int(y_work[0][0][di]) == 0:
                break
            decoder_input = y_work[:, :, di].squeeze(0)
            y_work = y_work.squeeze(0)
    else:
        y_work = y
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention, attention_sum = attn_decoder1(
                decoder_input, decoder_hidden, output_highfeature, output_area, attention_sum,
                decoder_attention, dense_input, batch_size, h_mask, w_mask, []
            )
            topv, topi = torch.max(decoder_output, 2)
            decoder_input = topi.view(batch_size)
            y_work = y_work.unsqueeze(0)
            for k in range(batch_size):
                if int(y_work[0][k][di]) == 0:
                    flag_z[k] += 1
                    if flag_z[k] > 1:
                        continue
                    else:
                        loss += criterion(decoder_output[k], y_work[:, k, di])
                else:
                    loss += criterion(decoder_output[k], y_work[:, k, di])
            y_work = y_work.squeeze(0)

    loss.backward()
    encoder_optimizer1.step()
    decoder_optimizer1.step()
    return loss.item()

# Build models and load pretrained encoder weights to CPU (if available)
encoder = densenet121()
pthfile = 'densenet121-a639ec97.pth'
if os.path.isfile(pthfile):
    try:
        pretrained_dict = torch.load(pthfile, map_location=device)
        encoder_dict = encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
        encoder_dict.update(pretrained_dict)
        encoder.load_state_dict(encoder_dict)
        print("Loaded pretrained densenet weights.")
    except Exception as e:
        print("Failed to load pretrained densenet:", e)
else:
    print(f"Pretrained file {pthfile} not found. Continuing with random init.")

attn_decoder1 = AttnDecoderRNN(hidden_size, 112, dropout_p=0.5)

# send to device (CPU)
encoder = encoder.to(device)
attn_decoder1 = attn_decoder1.to(device)

criterion = nn.NLLLoss()
decoder_input_init = torch.LongTensor([111] * batch_size).to(device)
decoder_hidden_init = torch.randn(batch_size, 1, hidden_size, device=device)
nn.init.xavier_uniform_(decoder_hidden_init)

# main training loop
num_epochs =1   # reduce this for quick tests
for epoch in range(num_epochs):
    encoder_optimizer1 = torch.optim.SGD(encoder.parameters(), lr=lr_rate, momentum=0.9)
    decoder_optimizer1 = torch.optim.SGD(attn_decoder1.parameters(), lr=lr_rate, momentum=0.9)

    running_loss = 0.0
    whole_loss = 0.0

    encoder.train()
    attn_decoder1.train()

    for step, (x, y) in enumerate(train_loader):
        if x.size(0) < batch_size:
            break

        # build masks h_mask,w_mask from the padding mask tensor
        h_mask = []
        w_mask = []
        for i in x:
            s_w = str(i[1][0])
            s_h = str(i[1][:, 1])
            w = s_w.count('1')
            h = s_h.count('1')
            h_comp = int(h / 16) + 1
            w_comp = int(w / 16) + 1
            h_mask.append(h_comp)
            w_mask.append(w_comp)

        x = x.to(device)
        y = y.to(device)
        output_highfeature = encoder(x)

        x_mean = [float(torch.mean(i)) for i in output_highfeature]
        for i in range(batch_size):
            decoder_hidden_init[i] = decoder_hidden_init[i] * x_mean[i]
            decoder_hidden_init[i] = torch.tanh(decoder_hidden_init[i])

        output_area1 = output_highfeature.size()
        output_area = output_area1[3]
        dense_input = output_area1[2]
        target_length = y.size(1)
        attention_sum_init = torch.zeros(batch_size, 1, dense_input, output_area, device=device)
        decoder_attention_init = torch.zeros(batch_size, 1, dense_input, output_area, device=device)

        running_loss += my_train(
            target_length, attn_decoder1, output_highfeature, output_area, y, criterion,
            encoder_optimizer1, decoder_optimizer1, x_mean, dense_input, h_mask, w_mask,
            decoder_input_init, decoder_hidden_init, attention_sum_init, decoder_attention_init
        )

        if step % 20 == 19:
            pre = ((step + 1) / max(1, len_train)) * 100 * batch_size
            whole_loss += running_loss
            running_loss = running_loss / (batch_size * 20)
            print(f'epoch {epoch}, lr {lr_rate:.5f}, te {teacher_forcing_ratio:.3f}, bs {batch_size}, loading {pre:.3f}%, running_loss {running_loss:f}')
            running_loss = 0.0

    loss_all_out = (whole_loss / max(1, len_train)) if len_train > 0 else 0.0
    print(f"epoch {epoch}, whole loss {loss_all_out:f}")

    # Evaluation
    total_dist = 0
    total_label = 0
    total_line = 0
    total_line_rec = 0

    encoder.eval()
    attn_decoder1.eval()
    print('Now, begin testing!!')

    for step_t, (x_t, y_t) in enumerate(test_loader):
        if x_t.size(0) < batch_size_t:
            break
        print(f'testing for {(step_t * 100 * batch_size_t / max(1, len_test)):.3f}%', end='\r')

        h_mask_t = []
        w_mask_t = []
        for i in x_t:
            s_w_t = str(i[1][0])
            s_h_t = str(i[1][:, 1])
            w_t = s_w_t.count('1')
            h_t = s_h_t.count('1')
            h_comp_t = int(h_t / 16) + 1
            w_comp_t = int(w_t / 16) + 1
            h_mask_t.append(h_comp_t)
            w_mask_t.append(w_comp_t)

        x_t = x_t.to(device)
        y_t = y_t.to(device)
        output_highfeature_t = encoder(x_t)

        x_mean_t = [float(torch.mean(i)) for i in output_highfeature_t]
        output_area_t1 = output_highfeature_t.size()
        output_area_t = output_area_t1[3]
        dense_input = output_area_t1[2]

        decoder_input_t = torch.LongTensor([111] * batch_size_t).to(device)
        decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size, device=device)
        nn.init.xavier_uniform_(decoder_hidden_t)

        for i in range(batch_size_t):
            decoder_hidden_t[i] = decoder_hidden_t[i] * x_mean_t[i]
            decoder_hidden_t[i] = torch.tanh(decoder_hidden_t[i])

        prediction = torch.zeros(batch_size_t, maxlen, dtype=torch.long)
        prediction_sub = []
        label_sub = []
        decoder_attention_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t, device=device)
        attention_sum_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t, device=device)

        m = torch.nn.ZeroPad2d((0, maxlen - y_t.size(1), 0, 0))
        y_t = m(y_t)

        for i in range(maxlen):
            decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(
                decoder_input_t, decoder_hidden_t, output_highfeature_t, output_area_t,
                attention_sum_t, decoder_attention_t, dense_input, batch_size_t, h_mask_t, w_mask_t, []
            )
            topv, topi = torch.max(decoder_output, 2)
            if torch.sum(topi) == 0:
                break
            decoder_input_t = topi.view(batch_size_t)
            prediction[:, i] = decoder_input_t

        for i in range(batch_size_t):
            for j in range(maxlen):
                if int(prediction[i][j]) == 0:
                    break
                else:
                    prediction_sub.append(int(prediction[i][j]))
            if len(prediction_sub) < maxlen:
                prediction_sub.append(0)

            for k in range(y_t.size(1)):
                if int(y_t[i][k]) == 0:
                    break
                else:
                    label_sub.append(int(y_t[i][k]))
            label_sub.append(0)

            dist, llen = cmp_result(label_sub, prediction_sub)
            total_dist += dist
            total_label += llen
            total_line += 1
            if dist == 0:
                total_line_rec += 1

            label_sub = []
            prediction_sub = []

    print('total_line_rec is', total_line_rec)
    wer = float(total_dist) / total_label if total_label > 0 else 0.0
    sacc = float(total_line_rec) / total_line if total_line > 0 else 0.0
    print(f'wer is {wer:.5f}')
    print(f'sacc is {sacc:.5f}')

    if sacc > exprate:
        exprate = sacc
        print(exprate)
        print("saving the model....")
        os.makedirs('model', exist_ok=True)
        torch.save(encoder.state_dict(), f'model/encoder_lr{lr_rate:.5f}_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl')
        torch.save(attn_decoder1.state_dict(), f'model/attn_decoder_lr{lr_rate:.5f}_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl')
        print("done")
        flag = 0
    else:
        flag += 1
        print(f'the best is {exprate:f}')
        print('the loss is bigger than before, so do not save the model')

    if flag == 10:
        lr_rate *= 0.1
        flag = 0