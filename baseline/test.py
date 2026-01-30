import os
import cv2
import torch
import numpy as np
import argparse
import torch.nn as nn
import torch.nn.functional as F
import shutil

# Assuming these imports work in your environment
from DRL.actor import *
from Renderer.stroke_gen import *
from Renderer.model import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
width = 128

parser = argparse.ArgumentParser(description='Learning to Paint')
parser.add_argument('--max_step', default=40, type=int, help='max length for episode')
parser.add_argument('--actor', default='./model/Paint-run1/actor.pkl', type=str, help='Actor model')
parser.add_argument('--renderer', default='./renderer.pkl', type=str, help='renderer model')
parser.add_argument('--img', default='image/test.png', type=str, help='test image')
parser.add_argument('--imgid', default=0, type=int, help='set begin number for generated image')
parser.add_argument('--divide', default=4, type=int, help='divide the target image to get better resolution')
# CHANGE 1: Set default saturation to 1.0 to keep colors "as they are" (Original)
parser.add_argument('--saturation', default=1.0, type=float, help='Saturation boost factor (1.0 = original colors)')
args = parser.parse_args()

canvas_cnt = args.divide * args.divide
T = torch.ones([1, 1, width, width], dtype=torch.float32).to(device)

# CHANGE 2: Convert Input from BGR (OpenCV) to RGB (Model)
# If we don't do this, the model thinks Red skin is Blue, and tries to paint it Blue.
img = cv2.imread(args.img, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

origin_shape = (img.shape[1], img.shape[0])

coord = torch.zeros([1, 2, width, width])
for i in range(width):
    for j in range(width):
        coord[0, 0, i, j] = i / (width - 1.)
        coord[0, 1, i, j] = j / (width - 1.)
coord = coord.to(device) 

Decoder = FCN()
Decoder.load_state_dict(torch.load(args.renderer))

def fix_color_vibrance(actions, factor):
    if factor == 1.0:
        return actions
    reshaped = actions.view(-1, 13).clone()
    rgb = reshaped[:, 10:13]
    gray = rgb.mean(dim=1, keepdim=True)
    rgb_fixed = gray + (rgb - gray) * factor
    rgb_fixed = torch.clamp(rgb_fixed, 0.0, 1.0)
    reshaped[:, 10:13] = rgb_fixed
    return reshaped.view(1, 65)

def decode(x, canvas): 
    x = x.view(-1, 10 + 3)
    stroke = 1 - Decoder(x[:, :10])
    stroke = stroke.view(-1, width, width, 1)
    color_stroke = stroke * x[:, -3:].view(-1, 1, 1, 3)
    stroke = stroke.permute(0, 3, 1, 2)
    color_stroke = color_stroke.permute(0, 3, 1, 2)
    stroke = stroke.view(-1, 5, 1, width, width)
    color_stroke = color_stroke.view(-1, 5, 3, width, width)
    res = []
    for i in range(5):
        canvas = canvas * (1 - stroke[:, i]) + color_stroke[:, i]
        res.append(canvas)
    return canvas, res

def small2large(x):
    x = x.reshape(args.divide, args.divide, width, width, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(args.divide * width, args.divide * width, -1)
    return x

def large2small(x):
    x = x.reshape(args.divide, width, args.divide, width, 3)
    x = np.transpose(x, (0, 2, 1, 3, 4))
    x = x.reshape(canvas_cnt, width, width, 3)
    return x

def smooth(img):
    def smooth_pix(img, tx, ty):
        if tx == args.divide * width - 1 or ty == args.divide * width - 1 or tx == 0 or ty == 0: 
            return img
        img[tx, ty] = (img[tx, ty] + img[tx + 1, ty] + img[tx, ty + 1] + img[tx - 1, ty] + img[tx, ty - 1] + img[tx + 1, ty - 1] + img[tx - 1, ty + 1] + img[tx - 1, ty - 1] + img[tx + 1, ty + 1]) / 9
        return img

    for p in range(args.divide):
        for q in range(args.divide):
            x = p * width
            y = q * width
            for k in range(width):
                img = smooth_pix(img, x + k, y + width - 1)
                if q != args.divide - 1:
                    img = smooth_pix(img, x + k, y + width)
            for k in range(width):
                img = smooth_pix(img, x + width - 1, y + k)
                if p != args.divide - 1:
                    img = smooth_pix(img, x + width, y + k)
    return img

def save_img(res, imgid, divide=False):
    output = res.detach().cpu().numpy()  
    output = np.transpose(output, (0, 2, 3, 1))
    if divide:
        output = small2large(output)
        output = smooth(output)
    else:
        output = output[0]
    
    output = (output * 255).astype('uint8')
    output = cv2.resize(output, origin_shape)
    
    # CHANGE 3: Convert Output from RGB (Model) to BGR (OpenCV)
    # The model creates RGB pixels, but cv2.imwrite expects BGR.
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite('output/generated' + str(imgid) + '.png', output)

actor = ResNet(9, 18, 65) 
actor.load_state_dict(torch.load(args.actor))
actor = actor.to(device).eval()
Decoder = Decoder.to(device).eval()

canvas = torch.zeros([1, 3, width, width]).to(device)

patch_img = cv2.resize(img, (width * args.divide, width * args.divide))
patch_img = large2small(patch_img)
patch_img = np.transpose(patch_img, (0, 3, 1, 2))
patch_img = torch.tensor(patch_img).to(device).float() / 255.

img = cv2.resize(img, (width, width))
img = img.reshape(1, width, width, 3)
img = np.transpose(img, (0, 3, 1, 2))
img = torch.tensor(img).to(device).float() / 255.

os.system('mkdir output')
os.makedirs("output_pts", exist_ok=True)
actions_over_time = []   

with torch.no_grad():
    if args.divide != 1:
        args.max_step = args.max_step // 2
    
    for i in range(args.max_step):
        stepnum = T * i / args.max_step
        actions = actor(torch.cat([canvas, img, stepnum, coord], 1))
        actions = fix_color_vibrance(actions, args.saturation)
        canvas, res = decode(actions, canvas)
        print('canvas step {}, L2Loss = {}'.format(i, ((canvas - img) ** 2).mean()))
        actions_over_time.append(actions.detach().cpu().squeeze(0))
        for j in range(5):
            save_img(res[j], args.imgid)
            args.imgid += 1

    if args.divide != 1:
        canvas = canvas[0].detach().cpu().numpy()
        canvas = np.transpose(canvas, (1, 2, 0))    
        canvas = cv2.resize(canvas, (width * args.divide, width * args.divide))
        canvas = large2small(canvas)
        canvas = np.transpose(canvas, (0, 3, 1, 2))
        canvas = torch.tensor(canvas).to(device).float()
        coord = coord.expand(canvas_cnt, 2, width, width)
        T = T.expand(canvas_cnt, 1, width, width)
        
        for i in range(args.max_step):
            stepnum = T * i / args.max_step
            actions = actor(torch.cat([canvas, patch_img, stepnum, coord], 1))
            actions = fix_color_vibrance(actions, args.saturation)
            canvas, res = decode(actions, canvas)
            print('divided canvas step {}, L2Loss = {}'.format(i, ((canvas - patch_img) ** 2).mean()))
            for j in range(5):
                save_img(res[j], args.imgid, True)
                args.imgid += 1

actions_stacked = torch.stack(actions_over_time, dim=0)

img_name = os.path.splitext(os.path.basename(args.img))[0]
torch.save(actions_stacked, f"output_pts/{img_name}.pt")
print(f"Saved actions for {img_name}: {actions_stacked.shape}")

final_img_dir = "output_100stroke_imgs"
os.makedirs(final_img_dir, exist_ok=True)
last_gen_path = f"output/generated{args.imgid - 1}.png"
out_name = f"{img_name}_generated.png"
shutil.copy2(last_gen_path, os.path.join(final_img_dir, out_name))
print(f"Saved final image -> {final_img_dir}/{out_name}")