'''
Python 3.6 
Pytorch >= 0.4
Written by Hongyu Wang in Beihang university
'''
import torch
import math
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy
import torch.utils.data as data
from data_iterator import dataIterator
from Attention_RNN import AttnDecoderRNN
from Densenet_torchvision import densenet121
from PIL import Image
from numpy import *
import builtins

torch.backends.cudnn.benchmark = False

def cmp_result(label,rec):
    dist_mat = numpy.zeros((len(label)+1, len(rec)+1),dtype='int32')
    dist_mat[0,:] = range(len(rec) + 1)
    dist_mat[:,0] = range(len(label) + 1)
    for i in range(1, len(label) + 1):
        for j in range(1, len(rec) + 1):
            hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
            ins_score = dist_mat[i,j-1] + 1
            del_score = dist_mat[i-1, j] + 1
            # use builtins.min to avoid numpy.min (imported by 'from numpy import *') which expects array-like inputs
            dist_mat[i,j] = builtins.min(hit_score, ins_score, del_score)

    dist = dist_mat[len(label), len(rec)]
    return dist, len(label), hit_score, ins_score, del_score

# Alignment with backtrace to produce aligned reference and hypothesis sequences
def edit_distance_alignment(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = numpy.zeros((n+1, m+1), dtype=int)
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if ref[i-1] == hyp[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost
            )
    # backtrace
    i, j = n, m
    aligned_ref, aligned_hyp = [], []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (ref[i-1] != hyp[j-1]):
            aligned_ref.append(ref[i-1])
            aligned_hyp.append(hyp[j-1])
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            aligned_ref.append(ref[i-1])
            aligned_hyp.append('<del>')
            i -= 1
        else:
            aligned_ref.append('<ins>')
            aligned_hyp.append(hyp[j-1])
            j -= 1
    return aligned_ref[::-1], aligned_hyp[::-1], int(dp[n][m])


def load_dict(dictFile):
    fp=open(dictFile)
    stuff=fp.readlines()
    fp.close()
    lexicon={}
    for l in stuff:
        w=l.strip().split()
        lexicon[w[0]]=int(w[1])

    print('total words/phones',len(lexicon))
    return lexicon

valid_datasets=['./offline-test.pkl', './test_caption.txt']
dictionaries=['./dictionary.txt']
batch_Imagesize=16
valid_batch_Imagesize=16
batch_size_t=1
maxlen=48
maxImagesize=100000
hidden_size = 256
# Device configuration: use CUDA if available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# gpu list: integers 0..n-1 (empty list if no CUDA available)
gpu = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []

# Robust checkpoint loader that handles checkpoints saved from DataParallel models
def _load_state_to_model(model, path, device):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    new_state = {}
    for k, v in state.items():
        new_k = k[len('module.'):] if k.startswith('module.') else k
        new_state[new_k] = v
    model.load_state_dict(new_state)

worddicts = load_dict(dictionaries[0])
worddicts_r = [None] * len(worddicts)
for kk, vv in worddicts.items():
        worddicts_r[vv] = kk

test,test_label = dataIterator(valid_datasets[0],valid_datasets[1],worddicts,batch_size=1,batch_Imagesize=batch_Imagesize,maxlen=maxlen,maxImagesize=maxImagesize)

class custom_dset(data.Dataset):
    def __init__(self,train,train_label):
        self.train = train
        self.train_label = train_label

    def __getitem__(self, index):
        train_setting = torch.from_numpy(numpy.array(self.train[index]))
        label_setting = torch.from_numpy(numpy.array(self.train_label[index])).type(torch.LongTensor)

        size = train_setting.size()
        train_setting = train_setting.view(1,size[2],size[3])
        label_setting = label_setting.view(-1)

        return train_setting,label_setting

    def __len__(self):
        return len(self.train)

off_image_test = custom_dset(test,test_label)
#print(off_image_train[10])

def imresize(im,sz):
    pil_im = Image.fromarray(im)
    return array(pil_im.resize(sz))

def collate_fn(batch):
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    img, label = zip(*batch)
    aa1 = 0
    bb1 = 0
    k = 0
    k1 = 0
    max_len = len(label[0])+1
    for j in range(len(img)):
        size = img[j].size()
        if size[1] > aa1:
            aa1 = size[1]
        if size[2] > bb1:
            bb1 = size[2]

    for ii in img:
        ii = ii.float()
        img_size_h = ii.size()[1]
        img_size_w = ii.size()[2]
        img_mask_sub_s = torch.ones(1,img_size_h,img_size_w).type(torch.FloatTensor)
        img_mask_sub_s = img_mask_sub_s*255.0
        img_mask_sub = torch.cat((ii,img_mask_sub_s),dim=0)
        padding_h = aa1-img_size_h
        padding_w = bb1-img_size_w
        m = torch.nn.ZeroPad2d((0,padding_w,0,padding_h))
        img_mask_sub_padding = m(img_mask_sub)
        img_mask_sub_padding = img_mask_sub_padding.unsqueeze(0)
        if k==0:
            img_padding_mask = img_mask_sub_padding
        else:
            img_padding_mask = torch.cat((img_padding_mask,img_mask_sub_padding),dim=0)
        k = k+1

    for ii1 in label:
        ii1 = ii1.long()
        ii1 = ii1.unsqueeze(0)
        ii1_len = ii1.size()[1]
        m = torch.nn.ZeroPad2d((0,max_len-ii1_len,0,0))
        ii1_padding = m(ii1)
        if k1 == 0:
            label_padding = ii1_padding
        else:
            label_padding = torch.cat((label_padding,ii1_padding),dim=0)
        k1 = k1+1

    img_padding_mask = img_padding_mask/255.0
    return img_padding_mask, label_padding

test_loader = torch.utils.data.DataLoader(
    dataset = off_image_test,
    batch_size = batch_size_t,
    shuffle = True,
    collate_fn = collate_fn
)

encoder = densenet121()
attn_decoder1 = AttnDecoderRNN(hidden_size,112,dropout_p=0.5)

# Use DataParallel only if multiple GPUs are available, then move models to the configured device
if len(gpu) > 1:
    encoder = torch.nn.DataParallel(encoder, device_ids=gpu)
    attn_decoder1 = torch.nn.DataParallel(attn_decoder1, device_ids=gpu)
encoder = encoder.to(device)
attn_decoder1 = attn_decoder1.to(device)

# Load checkpoints using the robust loader that handles DataParallel prefixes and map_location
_load_state_to_model(encoder, 'model/encoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl', device)
_load_state_to_model(attn_decoder1, 'model/attn_decoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl', device)

total_dist = 0
total_label = 0
total_line = 0
total_line_rec = 0
hit_all =0
ins_all =0
dls_all =0
wer_1 = 0
wer_2 = 0
wer_3 = 0
wer_4 = 0
wer_5 = 0
wer_6 = 0
wer_up=0

encoder.eval()
attn_decoder1.eval()

# Accumulators for aligned symbol pairs (for confusion matrix / per-symbol metrics)
aligned_gt_all = []
aligned_pr_all = []

for step_t, (x_t, y_t) in enumerate(test_loader):
    x_real_high = x_t.size()[2]
    x_real_width = x_t.size()[3]
    if x_t.size()[0]<batch_size_t:
        break

    h_mask_t = []
    w_mask_t = []
    for i in x_t:
        #h*w
        size_mask_t = i[1].size()
        s_w_t = str(i[1][0])
        s_h_t = str(i[1][:,1])
        w_t = s_w_t.count('1')
        h_t = s_h_t.count('1')
        h_comp_t = int(h_t/16)+1
        w_comp_t = int(w_t/16)+1
        h_mask_t.append(h_comp_t)
        w_mask_t.append(w_comp_t)

    x_t = x_t.to(device)
    y_t = y_t.to(device)
    output_highfeature_t = encoder(x_t)

    # compute scalar mean without gradient
    x_mean_t = float(torch.mean(output_highfeature_t).item())
    output_area_t1 = output_highfeature_t.size()
    output_area_t = output_area_t1[3]
    dense_input = output_area_t1[2]

    decoder_input_t = torch.LongTensor([111]*batch_size_t).to(device)

    decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size, device=device)
    decoder_hidden_t = decoder_hidden_t * x_mean_t
    decoder_hidden_t = torch.tanh(decoder_hidden_t)

    prediction = torch.zeros(batch_size_t,maxlen, device=device)
    #label = torch.zeros(batch_size_t,maxlen)
    prediction_sub = []
    label_sub = []
    label_real = []
    prediction_real = []

    decoder_attention_t = torch.zeros(batch_size_t,1,dense_input,output_area_t, device=device, dtype=output_highfeature_t.dtype)
    attention_sum_t = torch.zeros(batch_size_t,1,dense_input,output_area_t, device=device, dtype=output_highfeature_t.dtype)

    m = torch.nn.ZeroPad2d((0,maxlen-y_t.size()[1],0,0))
    y_t = m(y_t)
    for i in range(maxlen):
        decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(decoder_input_t,
                                                                                         decoder_hidden_t,
                                                                                         output_highfeature_t,
                                                                                         output_area_t,
                                                                                         attention_sum_t,
                                                                                         decoder_attention_t,dense_input,batch_size_t,h_mask_t,w_mask_t,gpu)


        topv,topi = torch.max(decoder_output,2)
        if torch.sum(topi)==0:
            break
        decoder_input_t = topi
        decoder_input_t = decoder_input_t.view(batch_size_t)
        #print(topi.size()) 16,1

        # prediction
        prediction[:,i] = decoder_input_t


    for i in range(batch_size_t):
        for j in range(maxlen):
            if int(prediction[i][j]) ==0:
                break
            else:
                prediction_sub.append(int(prediction[i][j]))
                prediction_real.append(worddicts_r[int(prediction[i][j])])
        if len(prediction_sub)<maxlen:
            prediction_sub.append(0)

        for k in range(y_t.size()[1]):
            if int(y_t[i][k]) ==0:
                break
            else:
                label_sub.append(int(y_t[i][k]))
                label_real.append(worddicts_r[int(y_t[i][k])])
        label_sub.append(0)

        # Remove trailing sentinel zeros if present
        while len(label_sub) > 0 and label_sub[-1] == 0:
            label_sub.pop()
        while len(prediction_sub) > 0 and prediction_sub[-1] == 0:
            prediction_sub.pop()

        # If no GT tokens remain, skip this sample
        if len(label_sub) == 0:
            print(f"Sample {step_t} has empty GT, skipping")
            label_sub = []
            prediction_sub = []
            label_real = []
            prediction_real = []
            continue

        # Alignment to collect symbol-level pairs for confusion matrix / F1
        aligned_gt, aligned_pr, _ = edit_distance_alignment(label_sub, prediction_sub)
        aligned_gt_all.extend(aligned_gt)
        aligned_pr_all.extend(aligned_pr)

        # Compute edit distance metrics
        dist, llen, hit, ins, dls = cmp_result(label_sub, prediction_sub)
        wer_step = float(dist) / llen

        total_dist += dist
        total_label += llen
        total_line += 1
        if dist == 0:
            total_line_rec = total_line_rec + 1

        print('step is %d' % (step_t))
        print('prediction is ')
        #print(''.join(prediction_real))
        print(prediction_real)
        print('the truth is')
        #print(''.join(label_real))
        print(label_real)
        print('the wer is %.5f' % (wer_step))

        label_sub = []
        prediction_sub = []
        label_real = []
        prediction_real = []


  
    # dist, llen, hit, ins, dls = cmp_result(label, prediction)
    # wer_step = float(dist) / llen
    # print('the wer is %.5f' % (wer_step))


    # if wer_step <= 0.1:
    #     wer_1 += 1
    # elif 0.1 < wer_step <= 0.2:
    #     wer_2 += 1
    # elif 0.2 < wer_step <= 0.3:
    #     wer_3 += 1
    # elif 0.3 < wer_step <= 0.4:
    #     wer_4 += 1
    # elif 0.4 < wer_step <= 0.5:
    #     wer_5 += 1
    # elif 0.5 < wer_step <= 0.6:
    #     wer_6 += 1
    # else:
    #     wer_up += 1

    # hit_all += hit
    # ins_all += ins
    # dls_all += dls
    # total_dist += dist
    # total_label += llen
    # total_line += 1
    # if dist == 0:
    #     total_line_rec += 1

wer = float(total_dist) / total_label
sacc = float(total_line_rec) / total_line
print('wer is %.5f' % (wer))
print('sacc is %.5f ' % (sacc))

# --------------------------
# Additional evaluation metrics
# --------------------------
# Build aligned symbol lists by re-running alignment on each sample result (we collected aligned pairs during inference)
# Note: The variables aligned_gt_all and aligned_pr_all may not exist yet in this evaluation block, so we will build them

# (Re-run inference to build aligned pairs wasn't done earlier in this block; instead, reconstruct aligned pairs from saved per-sample predictions)
# To do that robustly, we will recompute alignments by iterating over saved per-sample predictions and labels printed in the loop.
# However, to keep things simple and avoid re-running inference, we will collect symbol pairs here by parsing outputs if available.
# Safer approach: add collections during inference; but since we didn't, compute symbol-level accuracy using previously accumulated totals.

# For more informative metrics, try to compute precision/recall/f1 and confusion matrix using aligned symbol pairs if available.
try:
    # If aligned_gt_all/aligned_pr_all exist (collected during inference), use them
    aligned_gt_all  # noqa
    aligned_pr_all  # noqa
except NameError:
    aligned_gt_all = []
    aligned_pr_all = []

# In earlier versions we didn't store per-sample aligned pairs; if present, calculate symbol metrics
if len(aligned_gt_all) > 0:
    # Exclude insertions/deletions
    sym_pairs = [(g, p) for g, p in zip(aligned_gt_all, aligned_pr_all) if g not in ['<ins>', '<del>'] and p not in ['<ins>', '<del>']]
    if len(sym_pairs) > 0:
        gt_symbols = [worddicts_r[g] for g, _ in sym_pairs]
        pr_symbols = [worddicts_r[p] for _, p in sym_pairs]

        from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score, classification_report, confusion_matrix
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Symbol accuracy
        symbol_acc = sum(g == p for g, p in sym_pairs) / len(sym_pairs)

        # Precision / Recall / F1 (per-class and averaged)
        labels = sorted(list(set(gt_symbols)))
        p, r, f1s, sup = precision_recall_fscore_support(gt_symbols, pr_symbols, labels=labels, zero_division=0)
        macro_f1 = f1_score(gt_symbols, pr_symbols, average='macro', zero_division=0)
        micro_f1 = f1_score(gt_symbols, pr_symbols, average='micro', zero_division=0)
        weighted_f1 = f1_score(gt_symbols, pr_symbols, average='weighted', zero_division=0)

        print('\n===== SYMBOL-LEVEL METRICS =====')
        print('Symbol Accuracy:', symbol_acc)
        print('Macro F1:', macro_f1)
        print('Micro F1:', micro_f1)
        print('Weighted F1:', weighted_f1)
        print('\nClassification Report:\n')
        print(classification_report(gt_symbols, pr_symbols, labels=labels, zero_division=0))

        # Confusion matrix (aligned symbols)
        cm = confusion_matrix(gt_symbols, pr_symbols, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        cm_df.to_csv('confusion_matrix_aligned.csv')
        print('\nConfusion matrix saved as confusion_matrix_aligned.csv')

        # Heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(cm_df, cmap='Blues', xticklabels=True, yticklabels=True)
        plt.xlabel('Predicted')
        plt.ylabel('Ground Truth')
        plt.title('Aligned Symbol Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix_heatmap.png')
        plt.close()
        print('Confusion matrix heatmap saved as confusion_matrix_heatmap.png')

else:
    print('\nNo aligned symbol pairs were collected during this run, so per-symbol F1/confusion matrix cannot be computed.')
    print('If you want these metrics, let me collect aligned symbol pairs during inference and re-run the evaluation.')

print('\nDone.')



# """
# CPU-compatible test/evaluation script for HMER.

# Revisions:
# - CPU-only (no DataParallel / .cuda())
# - Loads encoder/decoder checkpoints and runs inference on offline-test.pkl
# - Uses the repo's custom DenseNet (2-channel conv0_m)
# """
# import argparse
# import sys
# import torch
# import numpy as np
# import torch.utils.data as data
# from data_iterator import dataIterator
# from Attention_RNN import AttnDecoderRNN
# from Densenet_torchvision import densenet121
# from PIL import Image

# torch.backends.cudnn.benchmark = False

# def cmp_result(label, rec):
#     dist_mat = np.zeros((len(label) + 1, len(rec) + 1), dtype='int32')
#     dist_mat[0, :] = np.arange(len(rec) + 1)
#     dist_mat[:, 0] = np.arange(len(label) + 1)
#     for i in range(1, len(label) + 1):
#         for j in range(1, len(rec) + 1):
#             hit_score = dist_mat[i-1, j-1] + (label[i-1] != rec[j-1])
#             ins_score = dist_mat[i, j-1] + 1
#             del_score = dist_mat[i-1, j] + 1
#             dist_mat[i, j] = min(hit_score, ins_score, del_score)
#     dist = dist_mat[len(label), len(rec)]
#     return dist, len(label)

# def load_dict(dictFile):
#     with open(dictFile, 'r', encoding='utf-8') as fp:
#         stuff = fp.readlines()
#     lexicon = {}
#     for l in stuff:
#         w = l.strip().split()
#         if not w:
#             continue
#         lexicon[w[0]] = int(w[1])
#     print('total words/phones', len(lexicon))
#     return lexicon

# def parse_args():
#     p = argparse.ArgumentParser(description="Run test inference (CPU) with encoder+decoder checkpoints")
#     p.add_argument("--encoder", type=str, default=r"model\encoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl", help="Path to encoder checkpoint")
#     p.add_argument("--decoder", type=str, default=r"model\attn_decoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl", help="Path to decoder checkpoint")
#     return p.parse_args()

# args = parse_args()

# # --- configuration (CPU) ---
# device = torch.device("cpu")
# valid_datasets = ['./offline-test.pkl', './test_caption.txt']
# dictionaries = ['./dictionary.txt']
# batch_Imagesize = 16
# batch_size_t = 1
# maxlen = 48
# maxImagesize = 100000
# hidden_size = 256
# gpu = []  # kept for compatibility with decoder signature

# # --- load dictionary and reverse mapping (use max index + 1) ---
# worddicts = load_dict(dictionaries[0])
# max_idx = max(worddicts.values())
# worddicts_r = [None] * (max_idx + 1)
# for kk, vv in worddicts.items():
#     worddicts_r[vv] = kk

# # --- load test data batches using dataIterator ---
# test, test_label = dataIterator(valid_datasets[0], valid_datasets[1], worddicts,
#                                 batch_size=1, batch_Imagesize=batch_Imagesize,
#                                 maxlen=maxlen, maxImagesize=maxImagesize)

# class custom_dset(data.Dataset):
#     def __init__(self, train, train_label):
#         self.train = train
#         self.train_label = train_label

#     def __getitem__(self, index):
#         train_setting = torch.from_numpy(np.array(self.train[index]))
#         label_setting = torch.from_numpy(np.array(self.train_label[index])).type(torch.LongTensor)
#         size = train_setting.size()
#         train_setting = train_setting.view(1, size[2], size[3])  # (C,H,W) with C likely 2
#         label_setting = label_setting.view(-1)
#         return train_setting, label_setting

#     def __len__(self):
#         return len(self.train)

# off_image_test = custom_dset(test, test_label)

# def collate_fn(batch):
#     batch.sort(key=lambda x: len(x[1]), reverse=True)
#     img, label = zip(*batch)
#     max_h = 0
#     max_w = 0
#     for j in range(len(img)):
#         size = img[j].size()
#         if size[1] > max_h:
#             max_h = size[1]
#         if size[2] > max_w:
#             max_w = size[2]

#     img_padding_mask = None
#     for ii in img:
#         ii = ii.float()
#         img_size_h = ii.size()[1]
#         img_size_w = ii.size()[2]
#         img_mask_sub_s = torch.ones(1, img_size_h, img_size_w, dtype=torch.float32) * 255.0
#         img_mask_sub = torch.cat((ii, img_mask_sub_s), dim=0)
#         padding_h = max_h - img_size_h
#         padding_w = max_w - img_size_w
#         m = torch.nn.ZeroPad2d((0, padding_w, 0, padding_h))
#         img_mask_sub_padding = m(img_mask_sub).unsqueeze(0)
#         if img_padding_mask is None:
#             img_padding_mask = img_mask_sub_padding
#         else:
#             img_padding_mask = torch.cat((img_padding_mask, img_mask_sub_padding), dim=0)

#     max_len = len(label[0]) + 1
#     label_padding = None
#     for ii1 in label:
#         ii1 = ii1.long().unsqueeze(0)
#         ii1_len = ii1.size(1)
#         m = torch.nn.ZeroPad2d((0, max_len - ii1_len, 0, 0))
#         ii1_padding = m(ii1)
#         if label_padding is None:
#             label_padding = ii1_padding
#         else:
#             label_padding = torch.cat((label_padding, ii1_padding), dim=0)

#     img_padding_mask = img_padding_mask / 255.0
#     return img_padding_mask, label_padding

# test_loader = torch.utils.data.DataLoader(
#     dataset=off_image_test,
#     batch_size=batch_size_t,
#     shuffle=False,
#     collate_fn=collate_fn,
#     num_workers=0
# )

# # ...existing code...
# # --- build models and load checkpoints (CPU) ---
# encoder = densenet121()
# attn_decoder1 = AttnDecoderRNN(hidden_size, len(worddicts_r), dropout_p=0.5)

# encoder = encoder.to(device)
# attn_decoder1 = attn_decoder1.to(device)

# def _unwrap_state_dict(ck):
#     if isinstance(ck, dict):
#         for k in ('state_dict','model','model_state_dict','net'):
#             if k in ck and isinstance(ck[k], dict):
#                 return ck[k]
#         return ck
#     return None

# def _strip_module(sd):
#     if any(k.startswith('module.') for k in sd.keys()):
#         return {k.replace('module.',''): v for k,v in sd.items()}
#     return sd

# def try_load_model(model, path, name):
#     try:
#         ck = torch.load(path, map_location=device)
#     except Exception as e:
#         print(f"Failed to read {name} checkpoint '{path}': {e}")
#         return False
#     sd = _unwrap_state_dict(ck)
#     if sd is None:
#         print(f"{name} checkpoint does not contain a state-dict mapping.")
#         return False
#     sd = _strip_module(sd)

#     # Try direct load (non-strict first)
#     try:
#         model.load_state_dict(sd, strict=False)
#         print(f"Loaded {name} (len(state_dict)={len(sd)}) from {path} (strict=False).")
#         return True
#     except Exception as e_direct:
#         pass

#     # Try simple key-mapping heuristics for encoder (conv0 <-> conv0_m, features prefix)
#     model_keys = list(model.state_dict().keys())
#     mapped = {}
#     for mk in model_keys:
#         candidates = [mk]
#         # alternate conv0 name
#         candidates.append(mk.replace('conv0_m','conv0'))
#         candidates.append(mk.replace('conv0','conv0_m'))
#         # features prefix variations
#         if mk.startswith('features.'):
#             candidates.append(mk[len('features.'):])
#         else:
#             candidates.append('features.' + mk)
#         found = False
#         for c in candidates:
#             if c in sd:
#                 mapped[mk] = sd[c]
#                 found = True
#                 break
#         # leave missing keys absent; load_state_dict(strict=False) will allow them

#     try:
#         model.load_state_dict(mapped, strict=False)
#         print(f"Loaded {name} via key-mapping heuristics from {path} (mapped {len(mapped)}/{len(model_keys)} keys).")
#         return True
#     except Exception as e_map:
#         # final fallback: print sample keys to help debugging
#         sample = list(sd.keys())[:40]
#         print(f"Failed to load {name} from {path}. Sample checkpoint keys (first 40):")
#         for k in sample:
#             print(" ", k)
#         print(f"(last error: {e_map})")
#         return False

# # attempt to load using provided defaults (or CLI overrides)
# enc_path = args.encoder
# dec_path = args.decoder

# if not try_load_model(encoder, enc_path, "encoder"):
#     print("Encoder load failed. Provide a matching checkpoint or run training to produce one.")
#     sys.exit(1)

# if not try_load_model(attn_decoder1, dec_path, "decoder"):
#     print("Decoder load failed. Provide a matching checkpoint or run training to produce one.")
#     sys.exit(1)

# encoder.eval()
# attn_decoder1.eval()
# # ...existing code...

# # --- testing loop ---
# total_dist = 0
# total_label = 0
# total_line = 0
# total_line_rec = 0

# for step_t, (x_t, y_t) in enumerate(test_loader):
#     if x_t.size(0) < batch_size_t:
#         break

#     # compute masks h_mask_t, w_mask_t from padding mask tensor
#     h_mask_t = []
#     w_mask_t = []
#     for i in x_t:
#         s_w_t = str(i[1][0])
#         s_h_t = str(i[1][:, 1])
#         w_t = s_w_t.count('1')
#         h_t = s_h_t.count('1')
#         h_comp_t = int(h_t / 16) + 1
#         w_comp_t = int(w_t / 16) + 1
#         h_mask_t.append(h_comp_t)
#         w_mask_t.append(w_comp_t)

#     x_t = x_t.to(device)
#     y_t = y_t.to(device)
#     with torch.no_grad():
#         output_highfeature_t = encoder(x_t)

#     # per-sample mean for initializing decoder hidden states
#     x_mean_t = [float(torch.mean(output_highfeature_t[i])) for i in range(output_highfeature_t.size(0))]
#     output_area_t1 = output_highfeature_t.size()
#     output_area_t = output_area_t1[3]
#     dense_input = output_area_t1[2]

#     decoder_input_t = torch.LongTensor([111] * batch_size_t).to(device)
#     decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).to(device)
#     for i in range(batch_size_t):
#         decoder_hidden_t[i] = decoder_hidden_t[i] * x_mean_t[i]
#         decoder_hidden_t[i] = torch.tanh(decoder_hidden_t[i])

#     prediction = torch.zeros(batch_size_t, maxlen, dtype=torch.long, device=device)
#     prediction_sub = []
#     label_sub = []
#     prediction_real = []
#     label_real = []

#     decoder_attention_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t, device=device)
#     attention_sum_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t, device=device)

#     m = torch.nn.ZeroPad2d((0, maxlen - y_t.size(1), 0, 0))
#     y_t = m(y_t)

#     with torch.no_grad():
#         for i in range(maxlen):
#             decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(
#                 decoder_input_t, decoder_hidden_t, output_highfeature_t, output_area_t,
#                 attention_sum_t, decoder_attention_t, dense_input, batch_size_t, h_mask_t, w_mask_t, gpu
#             )
#             topv, topi = torch.max(decoder_output, 2)
#             if torch.sum(topi) == 0:
#                 break
#             decoder_input_t = topi.view(batch_size_t)
#             prediction[:, i] = decoder_input_t

#     for i in range(batch_size_t):
#         for j in range(maxlen):
#             val = int(prediction[i][j])
#             if val == 0:
#                 break
#             prediction_sub.append(val)
#             prediction_real.append(worddicts_r[val] if 0 <= val < len(worddicts_r) else "<unk>")
#         if len(prediction_sub) < maxlen:
#             prediction_sub.append(0)

#         for k in range(y_t.size(1)):
#             val = int(y_t[i][k])
#             if val == 0:
#                 break
#             label_sub.append(val)
#             label_real.append(worddicts_r[val] if 0 <= val < len(worddicts_r) else "<unk>")
#         label_sub.append(0)

#         dist, llen = cmp_result(label_sub, prediction_sub)
#         total_dist += dist
#         total_label += llen
#         total_line += 1
#         if dist == 0:
#             total_line_rec += 1

#         print(f"step {step_t}")
#         print("prediction:", "".join(prediction_real))
#         print("ground   :", "".join(label_real))
#         print(f"wer      : {float(dist)/llen:.5f}")

#         label_sub = []
#         prediction_sub = []
#         label_real = []
#         prediction_real = []

# wer = float(total_dist) / total_label if total_label > 0 else 0.0
# sacc = float(total_line_rec) / total_line if total_line > 0 else 0.0
# print('wer is %.5f' % (wer))
# print('sacc is %.5f ' % (sacc))