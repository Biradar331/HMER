from tkinter import *
from tkinter.filedialog import askopenfilename, asksaveasfilename
from PIL import Image, ImageTk
from tkinter import messagebox
import numpy as np
import torch
from for_test_V20 import for_test
import matplotlib.pyplot as plt
import os

# Device (CPU)
device = torch.device("cpu")

# Globals
_flag_image_loaded = False
_prediction_string = ""
_pil_image = None

# ---------------- Helper Functions ---------------- #

def imresize(im, sz):
    pil_im = Image.fromarray(im)
    return np.array(pil_im.resize(sz))

def resize(w_box, h_box, pil_image):
    w, h = pil_image.size
    factor = min([1.0*w_box/w, 1.0*h_box/h])
    width = int(w*factor)
    height = int(h*factor)
    return pil_image.resize((width, height), Image.ANTIALIAS)

# ---------------- Core Functions ---------------- #

def choosepic():
    global _flag_image_loaded, _pil_image
    path_ = askopenfilename(filetypes=[("Image files", ("*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff")), ("All files","*.*")])
    if not path_:
        return
    try:
        _pil_image = Image.open(path_).convert('L')
    except Exception as e:
        messagebox.showerror(title="Error", message=f"Unable to open image: {e}")
        return
    tk_img = ImageTk.PhotoImage(resize(500, 350, _pil_image))
    img_label.config(image=tk_img)
    img_label.image = tk_img
    status_label.config(text=os.path.basename(path_))
    _flag_image_loaded = True

def _sanitize_prediction(pred_tokens):
    if pred_tokens is None:
        return ""
    out = []
    for t in pred_tokens:
        if t in ("<eol>", "<sos>", None, ""):
            if t == "<eol>":
                break
            continue
        out.append(str(t))
    cleaned = []
    prev = None
    for tok in out:
        if tok == prev:
            continue
        cleaned.append(tok)
        prev = tok
    return "".join(cleaned)

def trans1():
    global _pil_image, _prediction_string
    if not _flag_image_loaded:
        messagebox.showerror(title='Error', message='No Image')
        return

    status_label.config(text="Detecting...")
    status_label.update()

    arr = np.array(_pil_image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    try:
        attention, prediction = for_test(tensor, device=device)
    except Exception as e:
        messagebox.showerror(title="Inference error", message=str(e))
        status_label.config(text="Inference failed")
        return

    _prediction_string = _sanitize_prediction(prediction)
    result_label.config(text=_prediction_string)
    status_label.config(text="Done")

def trans2():
    global _prediction_string
    if not _flag_image_loaded:
        messagebox.showerror(title='Error', message='No Image')
        return
    if not _prediction_string:
        messagebox.showinfo(title='No prediction', message='No prediction to show')
        return

    def _balance_braces(s):
        opens = s.count('{')
        closes = s.count('}')
        if opens > closes:
            s = s + ('}' * (opens - closes))
        return s

    def _looks_like_safe_latex(s):
        # if contains raw backslash sequences (e.g. '\int', '\sqrt') rely on math rendering;
        # but if there are unknown backslash words (no letters after \) or many backslashes, avoid math parsing.
        if '\\' not in s:
            return True
        # if there are suspicious sequences like '\intcd' (unknown command) prefer plain text
        import re
        tokens = re.findall(r'\\[A-Za-z]+', s)
        # allow common math commands, otherwise decline
        allowed = {'frac','sqrt','sin','cos','tan','theta','alpha','beta','gamma','int','sum','le','ge','cdot','times','pm','leq','geq'}
        for t in tokens:
            if t.lstrip('\\') not in allowed:
                return False
        return True

    safe_str = _prediction_string
    safe_str = _balance_braces(safe_str)

    fig = plt.figure(figsize=(6,2.5))
    ax = fig.add_subplot(111)
    ax.axis('off')

    # if string looks unsafe for math parsing, render as plain text
    if not _looks_like_safe_latex(safe_str):
        ax.text(0.02, 0.6, safe_str, fontsize=18)
        plt.show()
        return

    # try math rendering, fallback to plain text on any exception
    try:
        ax.text(0.02, 0.6, f"${safe_str}$", fontsize=26)
        plt.show()
    except Exception:
        try:
            ax.clear()
            ax.axis('off')
            ax.text(0.02, 0.6, safe_str, fontsize=18)
            plt.show()
        except Exception as e:
            messagebox.showerror(title="Render error", message=f"Unable to render expression: {e}")

def saveClick():
    global _prediction_string
    if not _prediction_string:
        messagebox.showinfo(title='No prediction', message='Nothing to save')
        return
    path = asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt"),("All files","*.*")])
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(_prediction_string)
        messagebox.showinfo(title="Saved", message=f"Prediction saved to {path}")

# ---------------- GUI Layout ---------------- #

root = Tk()
root.title('Handwritten Math Expression Recognition Tool')
root.geometry('1200x800')
root.minsize(900, 600)
root.configure(bg="white")

# ---- Menu Bar ---- #
menubar = Menu(root)
filemenu = Menu(menubar, tearoff=0)
menubar.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Load', command=choosepic)
filemenu.add_command(label='Save', command=saveClick)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=root.quit)
root.config(menu=menubar)

# ---- Title ---- #
title = Label(root, text='Handwritten Mathmetical Expression Recognition System', 
              font=('Segoe UI', 24, 'bold'), bg="white", fg="#333")
title.pack(pady=15)

# ---- Button Frame (Flat + Responsive) ---- #
button_frame = Frame(root, bg="white")
button_frame.pack(fill=X, padx=20, pady=10)

btn_style = {'width': 15, 'height': 2, 'font': ('Segoe UI', 11, 'bold'), 'bg': "#f2f2f2", 'bd': 0, 'activebackground': "#dcdcdc"}

btn_load = Button(button_frame, text='Load Image', command=choosepic, **btn_style)
btn_detect = Button(button_frame, text='Start Detection', command=trans1, **btn_style)
btn_show = Button(button_frame, text='Show Formula', command=trans2, **btn_style)
btn_save = Button(button_frame, text='Save Prediction', command=saveClick, **btn_style)

# Grid layout for responsiveness
button_frame.columnconfigure((0, 1, 2, 3), weight=1)
btn_load.grid(row=0, column=0, padx=10, pady=5, sticky="ew")
btn_detect.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
btn_show.grid(row=0, column=2, padx=10, pady=5, sticky="ew")
btn_save.grid(row=0, column=3, padx=10, pady=5, sticky="ew")

# ---- Main Content Frame ---- #
main_frame = Frame(root, bg="white")
main_frame.pack(fill=BOTH, expand=True, padx=20, pady=10)
main_frame.columnconfigure(0, weight=3)
main_frame.columnconfigure(1, weight=2)
main_frame.rowconfigure(0, weight=1)

# Left side: Image
img_frame = LabelFrame(main_frame, text="Input Image", font=('Segoe UI', 12, 'bold'), bg="white", fg="#333")
img_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=5)

img_label = Label(img_frame, bg="white")
img_label.pack(fill=BOTH, expand=True, padx=10, pady=10)

# Right side: Prediction Output
output_frame = LabelFrame(main_frame, text="Prediction Result", font=('Segoe UI', 12, 'bold'), bg="white", fg="#333")
output_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=5)

status_label = Label(output_frame, text="No image loaded", anchor="w", bg="white", fg="#666", font=("Segoe UI", 10))
status_label.pack(fill=X, padx=10, pady=5)

result_label = Label(output_frame, text="", anchor="nw", justify="left", bg="white", fg="#111", font=("Consolas", 16), wraplength=400)
result_label.pack(fill=BOTH, expand=True, padx=10, pady=10)

# ---- Footer ---- #
footer = Label(root, text="Developed using PyTorch + Tkinter", bg="white", fg="#777", font=('Segoe UI', 9))
footer.pack(side=BOTTOM, pady=8)

root.mainloop()
