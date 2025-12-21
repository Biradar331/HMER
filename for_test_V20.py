import os
import torch
import torch.nn.functional as F
import numpy as np
from Densenet_torchvision import densenet121
from Attention_RNN import AttnDecoderRNN

# ============================================================
# CPU-only defaults
# ============================================================
device = torch.device("cpu")
hidden_size = 256
batch_size_t = 1
maxlen = 100

# paths (relative to project root)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DICT_PATH = os.path.join(BASE_DIR, "dictionary.txt")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "encoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl")
DECODER_PATH = os.path.join(BASE_DIR, "model", "attn_decoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl")


# ============================================================
# Load Dictionary
# ============================================================
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


# load vocabulary once
worddicts = load_dict(DICT_PATH)
worddicts_r = [None] * (max(worddicts.values()) + 1)
for kk, vv in worddicts.items():
    worddicts_r[vv] = kk


# ============================================================
# Model Utilities
# ============================================================
def _unwrap_state_dict(ck):
    if isinstance(ck, dict):
        for k in ('state_dict', 'model', 'model_state_dict', 'net'):
            if k in ck and isinstance(ck[k], dict):
                return ck[k]
        return ck
    return None


def _load_checkpoint_to_model(model, path):
    ck = torch.load(path, map_location=device)
    sd = _unwrap_state_dict(ck)
    if sd is None:
        raise RuntimeError("Checkpoint does not contain state-dict mapping")
    # strip 'module.' if present
    if any(k.startswith('module.') for k in sd.keys()):
        sd = {k.replace('module.', ''): v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)


# ============================================================
# Core Inference Function
# ============================================================
def for_test(x_t, device=device):
    """
    x_t: torch.Tensor shape [1,1,H,W] or [1,2,H,W] (float, not normalized to [0,1] expected)
    returns: attention_maps (numpy array), prediction_tokens (list of strings)
    """
    # build models
    encoder = densenet121()
    attn_decoder1 = AttnDecoderRNN(hidden_size, len(worddicts_r), dropout_p=0.5)

    encoder = encoder.to(device)
    attn_decoder1 = attn_decoder1.to(device)

    # load checkpoints
    _load_checkpoint_to_model(encoder, ENCODER_PATH)
    _load_checkpoint_to_model(attn_decoder1, DECODER_PATH)

    encoder.eval()
    attn_decoder1.eval()

    # prepare input tensor
    x_in = x_t.to(device).float()

    # if input has 1 channel, create mask channel as in training
    if x_in.dim() == 4 and x_in.size(1) == 1:
        mask = torch.ones_like(x_in)
        x_in = torch.cat((x_in, mask), dim=1)
    elif x_in.dim() == 4 and x_in.size(1) == 2:
        pass
    else:
        # reshape 2D to [1,1,H,W]
        if x_in.dim() == 2:
            x_in = x_in.unsqueeze(0).unsqueeze(0)
            mask = torch.ones_like(x_in)
            x_in = torch.cat((x_in, mask), dim=1)

    # compute features
    with torch.no_grad():
        output_highfeature_t = encoder(x_in)

    # per-sample mean for initializing decoder hidden states
    batch = output_highfeature_t.size(0)
    x_mean_t = [float(torch.mean(output_highfeature_t[i])) for i in range(batch)]
    output_area_t1 = output_highfeature_t.size()
    output_area_t = output_area_t1[3]
    dense_input = output_area_t1[2]

    # decoder init
    decoder_input_t = torch.LongTensor([111] * batch_size_t).to(device)
    decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).to(device)
    for i in range(batch_size_t):
        decoder_hidden_t[i] = decoder_hidden_t[i] * x_mean_t[i]
        decoder_hidden_t[i] = torch.tanh(decoder_hidden_t[i])

    prediction = torch.zeros(batch_size_t, maxlen, dtype=torch.long, device=device)
    decoder_attention_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t, device=device)
    attention_sum_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t, device=device)

    decoder_attention_t_cat = []

    with torch.no_grad():
        for i in range(maxlen):
            decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(
                decoder_input_t, decoder_hidden_t, output_highfeature_t, output_area_t,
                attention_sum_t, decoder_attention_t, dense_input, batch_size_t, [1], [1], []
            )

            # store attention
            decoder_attention_t_cat.append(decoder_attention_t[0].cpu().numpy())
            topv, topi = torch.max(decoder_output, 2)
            if torch.sum(topi) == 0:
                break
            decoder_input_t = topi.view(batch_size_t)
            prediction[:, i] = decoder_input_t

    attention_np = np.array(decoder_attention_t_cat)  # shape [T, 1, dense_input, output_area]
    pred_seq = prediction[0].cpu().numpy()
    prediction_real = []
    for val in pred_seq:
        if int(val) == 0:
            break
        idx = int(val)
        token = worddicts_r[idx] if 0 <= idx < len(worddicts_r) else "<unk>"
        prediction_real.append(token)
    prediction_real.append('<eol>')

    return attention_np, np.array(prediction_real)

if __name__ == "__main__":
    import cv2      

    TEST_DIR = os.path.join(BASE_DIR, "off_image_test")

    # Loop through all images in the test folder
    for img_name in os.listdir(TEST_DIR):
        if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            img_path = os.path.join(TEST_DIR, img_name)
            print(f"\nProcessing: {img_name}")

            # Load image in grayscale
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"❌ Failed to load {img_name}")
                continue

            # Convert to tensor [1,1,H,W]
            x_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()

            # Run model inference
            attention_maps, prediction_tokens = for_test(x_t)

            print("Predicted tokens:", prediction_tokens)



# import os
# import torch
# import torch.nn.functional as F
# import numpy as np
# from Densenet_torchvision import densenet121
# from Attention_RNN import AttnDecoderRNN
# import json
# import argparse
# import matplotlib.pyplot as plt
# from collections import Counter
# from data_iterator import dataIterator

# # CPU-only defaults
# device = torch.device("cpu")
# hidden_size = 256
# batch_size_t = 1
# maxlen = 100

# # paths (relative to project root)
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DICT_PATH = os.path.join(BASE_DIR, "dictionary.txt")
# ENCODER_PATH = os.path.join(BASE_DIR, "model", "encoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl")
# DECODER_PATH = os.path.join(BASE_DIR, "model", "attn_decoder_lr0.00001_GN_te1_d05_SGD_bs6_mask_conv_bn_b_xavier.pkl")

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

# # load vocabulary once
# worddicts = load_dict(DICT_PATH)
# worddicts_r = [None] * (max(worddicts.values()) + 1)
# for kk, vv in worddicts.items():
#     worddicts_r[vv] = kk

# def _unwrap_state_dict(ck):
#     if isinstance(ck, dict):
#         for k in ('state_dict','model','model_state_dict','net'):
#             if k in ck and isinstance(ck[k], dict):
#                 return ck[k]
#         return ck
#     return None

# def _load_checkpoint_to_model(model, path):
#     ck = torch.load(path, map_location=device)
#     sd = _unwrap_state_dict(ck)
#     if sd is None:
#         raise RuntimeError("Checkpoint does not contain state-dict mapping")
#     # strip 'module.' if present
#     if any(k.startswith('module.') for k in sd.keys()):
#         sd = {k.replace('module.',''): v for k,v in sd.items()}
#     model.load_state_dict(sd, strict=False)

# def for_test(x_t, device=device):
#     """
#     x_t: torch.Tensor shape [1,1,H,W] or [1,2,H,W] (float, not normalized to [0,1] expected)
#     returns: attention_maps (numpy array), prediction_tokens (list of strings)
#     """
#     # build models
#     encoder = densenet121()
#     attn_decoder1 = AttnDecoderRNN(hidden_size, len(worddicts_r), dropout_p=0.5)

#     encoder = encoder.to(device)
#     attn_decoder1 = attn_decoder1.to(device)

#     # load checkpoints (raise on error)
#     _load_checkpoint_to_model(encoder, ENCODER_PATH)
#     _load_checkpoint_to_model(attn_decoder1, DECODER_PATH)

#     encoder.eval()
#     attn_decoder1.eval()

#     # prepare input tensor on device
#     x_in = x_t.to(device).float()
#     # if input has 1 channel, create mask channel as in training (concatenate a channel of ones)
#     if x_in.dim() == 4 and x_in.size(1) == 1:
#         mask = torch.ones_like(x_in)
#         x_in = torch.cat((x_in, mask), dim=1)
#     elif x_in.dim() == 4 and x_in.size(1) == 2:
#         # assume already image+mask
#         pass
#     else:
#         # try to reshape 2D to [1,1,H,W]
#         if x_in.dim() == 2:
#             x_in = x_in.unsqueeze(0).unsqueeze(0)
#             mask = torch.ones_like(x_in)
#             x_in = torch.cat((x_in, mask), dim=1)

#     # compute features
#     with torch.no_grad():
#         output_highfeature_t = encoder(x_in)

#     # per-sample mean for initializing decoder hidden states
#     batch = output_highfeature_t.size(0)
#     x_mean_t = [float(torch.mean(output_highfeature_t[i])) for i in range(batch)]
#     output_area_t1 = output_highfeature_t.size()
#     output_area_t = output_area_t1[3]
#     dense_input = output_area_t1[2]

#     # decoder init
#     decoder_input_t = torch.LongTensor([111] * batch_size_t).to(device)
#     decoder_hidden_t = torch.randn(batch_size_t, 1, hidden_size).to(device)
#     for i in range(batch_size_t):
#         decoder_hidden_t[i] = decoder_hidden_t[i] * x_mean_t[i]
#         decoder_hidden_t[i] = torch.tanh(decoder_hidden_t[i])

#     prediction = torch.zeros(batch_size_t, maxlen, dtype=torch.long, device=device)
#     decoder_attention_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t, device=device)
#     attention_sum_t = torch.zeros(batch_size_t, 1, dense_input, output_area_t, device=device)

#     decoder_attention_t_cat = []

#     with torch.no_grad():
#         for i in range(maxlen):
#             decoder_output, decoder_hidden_t, decoder_attention_t, attention_sum_t = attn_decoder1(
#                 decoder_input_t, decoder_hidden_t, output_highfeature_t, output_area_t,
#                 attention_sum_t, decoder_attention_t, dense_input, batch_size_t, [1], [1], []
#             )
#             # store attention (cpu numpy)
#             decoder_attention_t_cat.append(decoder_attention_t[0].cpu().numpy())
#             topv, topi = torch.max(decoder_output, 2)
#             if torch.sum(topi) == 0:
#                 break
#             decoder_input_t = topi.view(batch_size_t)
#             prediction[:, i] = decoder_input_t

#     attention_np = np.array(decoder_attention_t_cat)  # shape [T, 1, dense_input, output_area]
#     pred_seq = prediction[0].cpu().numpy()
#     prediction_real = []
#     for val in pred_seq:
#         if int(val) == 0:
#             break
#         idx = int(val)
#         token = worddicts_r[idx] if 0 <= idx < len(worddicts_r) else "<unk>"
#         prediction_real.append(token)
#     prediction_real.append('<eol>')

#     return attention_np, np.array(prediction_real)



# def cmp_result(label_tokens, rec_tokens):
#     # Levenshtein distance for token lists
#     a = list(label_tokens)
#     b = list(rec_tokens)
#     la = len(a); lb = len(b)
#     if la == 0:
#         return lb, 0
#     dp = [[0] * (lb + 1) for _ in range(la + 1)]
#     for i in range(la + 1):
#         dp[i][0] = i
#     for j in range(lb + 1):
#         dp[0][j] = j
#     for i in range(1, la + 1):
#         for j in range(1, lb + 1):
#             cost = 0 if a[i-1] == b[j-1] else 1
#             dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
#     return dp[la][lb], la

# def tokens_from_label_array(label_arr):
#     """
#     Robust conversion of label container to token list.
#     Accepts: list, nested-list, numpy array, torch tensor.
#     Stops at first 0 sentinel. If elements are strings, they are returned directly.
#     """
#     # unwrap torch tensor
#     try:
#         import torch as _torch
#         if isinstance(label_arr, _torch.Tensor):
#             arr = label_arr.cpu().numpy()
#         else:
#             arr = label_arr
#     except Exception:
#         arr = label_arr

#     # convert nested list like [[...]] -> take first inner list if that looks right
#     if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], (list, tuple, np.ndarray)):
#         # pick the first non-empty inner sequence
#         inner = None
#         for candidate in arr:
#             if candidate:
#                 inner = candidate
#                 break
#         if inner is None:
#             arr = []
#         else:
#             arr = inner

#     # numpy-ify
#     try:
#         arr = np.array(arr)
#     except Exception:
#         # fallback: iterate as-is
#         arr = list(arr) if arr is not None else []

#     toks = []
#     # flatten 1D sequence
#     flat = arr.flatten() if isinstance(arr, np.ndarray) and arr.ndim > 0 else (arr if isinstance(arr, (list,tuple,np.ndarray)) else [arr])
#     for v in flat:
#         # if string token (already decoded), append directly
#         if isinstance(v, str):
#             if v == '<eol>':
#                 break
#             toks.append(v)
#             continue
#         # try numeric
#         try:
#             iv = int(v)
#         except Exception:
#             # skip unknown entries
#             continue
#         if iv == 0:
#             break
#         toks.append(worddicts_r[iv] if 0 <= iv < len(worddicts_r) else "<unk>")
#     return toks

# def _prepare_input_from_test_item(item_np):
#     """
#     Normalize a numpy array from dataIterator into a torch tensor of shape [1, C, H, W]
#     where C is 1 (image) or 2 (image + mask). Handles inputs that might already include
#     channel or batch dimensions.
#     """
#     arr = np.array(item_np)
#     # remove any extra singleton dimensions except channel
#     # possible shapes seen: (H,W), (1,H,W), (1,1,H,W), (C,H,W), (H,W,1) etc.
#     # We want (C,H,W) then add batch dim.
#     if arr.ndim == 2:
#         # (H,W) -> (1,H,W)
#         arr = arr[np.newaxis, ...]
#     elif arr.ndim == 3:
#         # Could be (1,H,W) or (H,W,1) or (C,H,W)
#         if arr.shape[0] == 1 or arr.shape[0] == 2:
#             # (C,H,W) good
#             pass
#         elif arr.shape[2] == 1:
#             # (H,W,1) -> (1,H,W)
#             arr = arr[..., 0][np.newaxis, ...]
#         else:
#             # ambiguous: treat as (H,W,C) -> transpose to (C,H,W)
#             arr = np.transpose(arr, (2,0,1))
#     elif arr.ndim == 4:
#         # Could be (1,C,H,W) or (C,1,H,W) — attempt to squeeze leading batch if present
#         if arr.shape[0] == 1 and (arr.shape[1] == 1 or arr.shape[1] == 2):
#             # already (1,C,H,W)
#             arr = arr[0]
#         elif arr.shape[1] == 1 and (arr.shape[0] == 1 or arr.shape[0] == 2):
#             # maybe (C,1,H,W) -> pick first dimension as channel
#             arr = arr[:,0,:,:]
#         else:
#             # flatten to (C,H,W) by squeezing singletons
#             arr = np.squeeze(arr)
#             if arr.ndim == 2:
#                 arr = arr[np.newaxis, ...]
#             elif arr.ndim == 3 and arr.shape[2] == 1:
#                 arr = arr[...,0][np.newaxis,...]
#     else:
#         # fallback: squeeze and re-evaluate
#         arr = np.squeeze(arr)
#         if arr.ndim == 2:
#             arr = arr[np.newaxis, ...]
#         elif arr.ndim == 3 and arr.shape[2] == 1:
#             arr = arr[...,0][np.newaxis,...]

#     # now arr should be (C,H,W)
#     if arr.ndim != 3:
#         raise RuntimeError(f"Unable to prepare input tensor, unexpected shape after normalization: {arr.shape}")

#     # convert to float32 and add batch dim -> (1,C,H,W)
#     t = torch.from_numpy(arr.astype(np.float32)).unsqueeze(0)
#     return t


# def evaluate_batch(dataset_pkl=None, captions_txt=None, max_samples=None, save_predictions=None, out_dir=None, plot=True, collapse_repeats=True):
#     """
#     Improved evaluation with plots and token confusion reporting.

#     Parameters:
#     - collapse_repeats: if True collapse consecutive repeated tokens in prediction
#       before computing edit distance (helps when model repeats one symbol many times).
#     """
#     if dataset_pkl is None:
#         dataset_pkl = os.path.join(BASE_DIR, "offline-test.pkl")
#     if captions_txt is None:
#         captions_txt = os.path.join(BASE_DIR, "test_caption.txt")
#     print("Loading test data:", dataset_pkl, captions_txt)
#     test, test_label = dataIterator(dataset_pkl, captions_txt, worddicts, batch_size=1, batch_Imagesize=16, maxlen=maxlen, maxImagesize=100000)
#     n = len(test)
#     if max_samples is not None:
#         n = min(n, int(max_samples))

#     total_dist = 0
#     total_label = 0
#     total_line = 0
#     correct_seq = 0
#     token_matches = 0
#     token_total = 0

#     wer_list = []
#     wer_list_raw = []
#     pred_len_list = []
#     preds = []

#     from collections import Counter
#     conf_counter = Counter()
#     true_counts = Counter()
#     correct_counts = Counter()

#     for i in range(n):
#         try:
#             img_np = np.array(test[i])
#             x_t = _prepare_input_from_test_item(img_np)
#         except Exception as e:
#             print(f"[{i}] Input preparation error: {e}")
#             continue

#         try:
#             attention, prediction_tokens = for_test(x_t)
#         except Exception as e:
#             print(f"[{i}] Inference error: {e}")
#             continue

#         # build prediction list (stop at <eol>)
#         pred_list = []
#         for t in prediction_tokens:
#             if t == '<eol>':
#                 break
#             pred_list.append(t)

#         # optional collapse of repeats (helps when model loops)
#         if collapse_repeats and len(pred_list) > 0:
#             pred_list_coll = [pred_list[0]]
#             for tok in pred_list[1:]:
#                 if tok != pred_list_coll[-1]:
#                     pred_list_coll.append(tok)
#             pred_for_eval = pred_list_coll
#         else:
#             pred_for_eval = pred_list

#         label_tokens = tokens_from_label_array(test_label[i])

#         # compute edit distance (raw) and normalized (capped at 1)
#         dist, llen = cmp_result(label_tokens, pred_for_eval)
#         wer_raw = float(dist) / llen if llen > 0 else 0.0
#         wer = min(wer_raw, 1.0)  # cap for reporting
#         wer_list.append(wer)
#         wer_list_raw.append(wer_raw)
#         pred_len_list.append(len(pred_for_eval))

#         total_dist += dist
#         total_label += llen
#         total_line += 1
#         if dist == 0:
#             correct_seq += 1

#         # token-level simple alignment statistics (left-to-right)
#         L = max(len(label_tokens), len(pred_for_eval))
#         for k in range(L):
#             if k < len(label_tokens):
#                 true_tok = label_tokens[k]
#                 true_counts[true_tok] += 1
#             else:
#                 true_tok = None
#             if k < len(pred_for_eval):
#                 pred_tok = pred_for_eval[k]
#             else:
#                 pred_tok = None
#             if true_tok is not None and pred_tok is not None:
#                 token_total += 1
#                 if true_tok == pred_tok:
#                     token_matches += 1
#                     correct_counts[true_tok] += 1
#                 else:
#                     conf_counter[(true_tok, pred_tok)] += 1

#         preds.append({
#             "index": i,
#             "label": "".join(label_tokens),
#             "pred": "".join(pred_for_eval),
#             "wer_raw": wer_raw,
#             "wer_capped": wer,
#             "pred_len": len(pred_for_eval)
#         })

#         if (i + 1) % 50 == 0 or i == n-1:
#             cur_wer = (total_dist / total_label) if total_label > 0 else 0.0
#             print(f"Processed {i+1}/{n}  current raw WER={(cur_wer):.5f}  sacc={(correct_seq/total_line):.5f}")

#     if total_line == 0:
#         print("No samples processed successfully. Exiting evaluation.")
#         return {"summary": {"samples": 0, "wer": 0.0, "sacc": 0.0, "token_acc": 0.0}, "preds": []}

#     wer_final_raw = float(total_dist) / total_label if total_label > 0 else 0.0
#     wer_final = min(wer_final_raw, 1.0)
#     sacc = float(correct_seq) / total_line if total_line > 0 else 0.0
#     token_acc = float(token_matches) / token_total if token_total > 0 else 0.0

#     summary = {
#         "samples": total_line,
#         "wer_raw": wer_final_raw,
#         "wer_capped": wer_final,
#         "sacc": sacc,
#         "token_acc": token_acc,
#         "preds_count": len(preds)
#     }

#     print("=== Evaluation Summary ===")
#     print(f"Samples evaluated: {total_line}")
#     print(f"WER (raw): {wer_final_raw:.5f}   WER (capped to 1.0): {wer_final:.5f}")
#     print(f"Sequence accuracy: {sacc:.5f}")
#     print(f"Token accuracy: {token_acc:.5f}")

#     # save predictions if requested
#     if save_predictions:
#         with open(save_predictions, "w", encoding="utf-8") as f:
#             json.dump({"summary": summary, "preds": preds}, f, indent=2, ensure_ascii=False)
#         print("Saved predictions to:", save_predictions)

#     # plots and reports
#     if plot and len(wer_list) > 0:
#         if out_dir and not os.path.exists(out_dir):
#             os.makedirs(out_dir)

#         # WER histogram (raw and capped)
#         plt.figure(figsize=(6,4))
#         plt.hist(wer_list_raw, bins=50, color='tab:blue', alpha=0.8)
#         plt.xlabel("Raw WER (edits / label length)")
#         plt.ylabel("Count")
#         plt.title("Raw WER distribution")
#         if out_dir:
#             p = os.path.join(out_dir, "wer_hist_raw.png"); plt.savefig(p, bbox_inches='tight'); print("Saved", p)
#         else:
#             plt.show()
#         plt.close()

#         # WER CDF (capped)
#         sorted_wer = np.sort(wer_list)
#         cdf = np.arange(1, len(sorted_wer)+1) / float(len(sorted_wer))
#         plt.figure(figsize=(6,4))
#         plt.plot(sorted_wer, cdf, marker='.', linestyle='none')
#         plt.xlabel("WER (capped at 1.0)")
#         plt.ylabel("CDF")
#         plt.title("WER CDF")
#         if out_dir:
#             p = os.path.join(out_dir, "wer_cdf.png"); plt.savefig(p, bbox_inches='tight'); print("Saved", p)
#         else:
#             plt.show()
#         plt.close()

#         # Prediction length distribution
#         bins = max(10, min(50, (max(pred_len_list) + 1) if len(pred_len_list)>0 else 10))
#         plt.figure(figsize=(6,4))
#         plt.hist(pred_len_list, bins=bins, color='tab:green', alpha=0.8)
#         plt.xlabel("Prediction length (tokens)")
#         plt.ylabel("Count")
#         plt.title("Prediction length distribution")
#         if out_dir:
#             p = os.path.join(out_dir, "pred_len_hist.png"); plt.savefig(p, bbox_inches='tight'); print("Saved", p)
#         else:
#             plt.show()
#         plt.close()

#         # Per-token accuracy (for tokens with support >= min_support)
#         min_support = 5
#         token_stats = []
#         for tok, total in true_counts.items():
#             correct = correct_counts.get(tok, 0)
#             acc = float(correct) / total if total > 0 else 0.0
#             token_stats.append((tok, total, acc))
#         token_stats_sorted = sorted(token_stats, key=lambda x: x[1], reverse=True)  # by support
#         top_tokens = token_stats_sorted[:30]
#         if len(top_tokens) > 0:
#             toks = [t for t, s, a in top_tokens]
#             supports = [s for t, s, a in top_tokens]
#             accs = [a for t, s, a in top_tokens]
#             plt.figure(figsize=(10,4))
#             plt.bar(range(len(toks)), accs, tick_label=toks, color='tab:orange')
#             plt.xticks(rotation=90)
#             plt.ylabel("Token accuracy")
#             plt.title("Top tokens by support: accuracy")
#             plt.tight_layout()
#             if out_dir:
#                 p = os.path.join(out_dir, "token_accuracy_top.png"); plt.savefig(p, bbox_inches='tight'); print("Saved", p)
#             else:
#                 plt.show()
#             plt.close()

#             # save per-token CSV
#             if out_dir:
#                 import csv
#                 csv_path = os.path.join(out_dir, "token_stats.csv")
#                 with open(csv_path, "w", newline="", encoding="utf-8") as cf:
#                     writer = csv.writer(cf)
#                     writer.writerow(["token","support","accuracy"])
#                     for tok,s,a in token_stats_sorted:
#                         writer.writerow([tok,s,f"{a:.6f}"])
#                 print("Saved token stats CSV to", csv_path)

#         # Confusion matrix for top N tokens by support
#         topN = 20
#         top_tokens_conf = [t for t,_,_ in token_stats_sorted[:topN]]
#         if len(top_tokens_conf) > 0:
#             mat = np.zeros((len(top_tokens_conf), len(top_tokens_conf)), dtype=np.int32)
#             idx_map = {t:i for i,t in enumerate(top_tokens_conf)}
#             for (tru, pred), cnt in conf_counter.items():
#                 if tru in idx_map and pred in idx_map:
#                     mat[idx_map[tru], idx_map[pred]] += cnt
#             # normalize rows for display
#             row_sums = mat.sum(axis=1, keepdims=True).astype(np.float32)
#             norm_mat = np.divide(mat, row_sums, out=np.zeros_like(mat, dtype=float), where=row_sums!=0)
#             plt.figure(figsize=(8,6))
#             im = plt.imshow(norm_mat, interpolation='nearest', cmap='viridis')
#             plt.colorbar(im, fraction=0.046, pad=0.04)
#             plt.xticks(range(len(top_tokens_conf)), top_tokens_conf, rotation=90)
#             plt.yticks(range(len(top_tokens_conf)), top_tokens_conf)
#             plt.xlabel("Predicted")
#             plt.ylabel("True")
#             plt.title("Confusion matrix (normalized rows) for top tokens")
#             plt.tight_layout()
#             if out_dir:
#                 p = os.path.join(out_dir, "confusion_top_tokens.png"); plt.savefig(p, bbox_inches='tight'); print("Saved", p)
#             else:
#                 plt.show()
#             plt.close()

#         # Top error CSV and annotated images
#         sorted_preds = sorted(preds, key=lambda x: x["wer_raw"], reverse=True)
#         top_errors = sorted_preds[:20]
#         if out_dir:
#             with open(os.path.join(out_dir, "top_errors.csv"), "w", newline="", encoding="utf-8") as cf:
#                 import csv
#                 writer = csv.writer(cf)
#                 writer.writerow(["index","label","pred","wer_raw","pred_len"])
#                 for e in top_errors:
#                     writer.writerow([e["index"], e["label"], e["pred"], f"{e['wer_raw']:.6f}", e["pred_len"]])
#             print("Saved top errors CSV to", os.path.join(out_dir, "top_errors.csv"))

#             # save small annotated images for top errors (best-effort)
#             try:
#                 from PIL import Image, ImageDraw, ImageFont
#                 for rank, e in enumerate(top_errors):
#                     idx = e["index"]
#                     img_arr = np.array(test[idx])
#                     if img_arr.ndim == 3 and img_arr.shape[0] in (1,2):
#                         img_disp = img_arr[0]
#                     elif img_arr.ndim == 4 and img_arr.shape[0] == 1 and img_arr.shape[1] in (1,2):
#                         img_disp = img_arr[0][0]
#                     elif img_arr.ndim == 2:
#                         img_disp = img_arr
#                     else:
#                         img_disp = np.squeeze(img_arr)
#                     # scale to 0..255
#                     if img_disp.max() <= 1.0:
#                         pil = Image.fromarray((img_disp * 255).astype(np.uint8))
#                     else:
#                         pil = Image.fromarray(img_disp.astype(np.uint8))
#                     draw = ImageDraw.Draw(pil)
#                     try:
#                         font = ImageFont.truetype("arial.ttf", 12)
#                     except Exception:
#                         font = ImageFont.load_default()
#                     header = f"idx:{idx} wer:{e['wer_raw']:.3f}"
#                     draw.rectangle([(0,0),(pil.size[0],36)], fill=(255))
#                     draw.text((4,2), header, fill=(0), font=font)
#                     draw.text((4,16), f"L:{e['label']}", fill=(0), font=font)
#                     draw.text((4,28), f"P:{e['pred']}", fill=(0), font=font)
#                     out_img = os.path.join(out_dir, f"top_error_{rank+1}_idx{idx}.png")
#                     pil.save(out_img)
#                 print("Saved annotated top-error images to", out_dir)
#             except Exception as _e:
#                 print("Failed to save top-error images:", _e)

#     return {"summary": summary, "preds": preds}
# # ...existing code...

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="for_test_V20 evaluation")
#     parser.add_argument("--mode", choices=["single","batch"], default="batch")
#     parser.add_argument("--image", type=str, help="single image path")
#     parser.add_argument("--dataset", type=str, help="offline-test.pkl path")
#     parser.add_argument("--captions", type=str, help="test captions file")
#     parser.add_argument("--max", type=int, help="max samples to evaluate")
#     parser.add_argument("--save", type=str, help="save predictions to JSON")
#     parser.add_argument("--out", type=str, help="output directory for plots")
#     args = parser.parse_args()

#     if args.mode == "single":
#         if not args.image:
#             print("Provide --image for single mode")
#         else:
#             im = Image.open(args.image).convert("L")
#             arr = np.array(im).astype(np.float32) / 255.0
#             if arr.ndim == 2:
#                 arr = arr[np.newaxis, ...]
#             x_t = torch.from_numpy(arr).unsqueeze(0)
#             attn, pred = for_test(x_t)
#             print("Prediction tokens:", pred)
#             print("Pred string:", "".join([t for t in pred if t != '<eol>']))
#     else:
#         evaluate_batch(dataset_pkl=args.dataset, captions_txt=args.captions, max_samples=args.max, save_predictions=args.save, out_dir=args.out, plot=True)
