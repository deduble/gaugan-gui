"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import socket
import threading
import os
import sys
import codecs
import argparse
import subprocess

from collections import OrderedDict
from flask import Flask, request
from base64 import b64decode, b64encode
import numpy as np
import cv2



parser = argparse.ArgumentParser(description=('This is an app that helps to USE '
                                              'Nvidia`s GAUGAN called SPADE with an interactive UI.'),
                                usage='python3 spade_app.py -s C:\\src\\SPADE -c C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe')


parser.add_argument('--spade-path', '-s', metavar="PATH/TO/SPADE", type=str,
                    help=("Absolute path to your Nvidia SPADE directory. Expects you installed SPADE per "
                         "directions"))

parser.add_argument('--chrome-path', '-c', metavar="PATH/TO/CHROME/BINARY", type=str,
                    default=r'C:\Program Files (x86)\Google\Chrome\Application\chrome.exe', 
                    help=("Path to your Chrome.exe binary."))

parser.add_argument('-debug', '-d', '--debug', action='store_false', help="Run flask app in debug mode")



args = parser.parse_args()
spade_path = args.spade_path
chrome_path = args.chrome_path
debug = args.debug

sys.path.append(spade_path)

if not spade_path:
    parser.print_help()
    exit()

# clean command line arguments because SPADE is using them
sys.argv = sys.argv[:1]

if os.path.exists(spade_path):
    import data
    from options.test_options import TestOptions
    from models.pix2pix_model import Pix2PixModel
    from util.visualizer import Visualizer
    from util import html
else:
    raise FileNotFoundError("Your SPADE directory is faulty.")

opt = TestOptions().parse()

opt.name = 'coco_pretrained'
opt.checkpoints_dir = os.path.join(spade_path, 'checkpoints')
custom_data_path = os.path.join(spade_path, 'custom_data')
opt.dataroot = custom_data_path

if not os.path.exists(custom_data_path):
    os.mkdir(custom_data_path)
    for validation_folder in ['val_img', 'val_inst', 'val_label']:
        os.makedirs(os.path.join(custom_data_path, validation_folder))

print(opt)

model = Pix2PixModel(opt)
model.eval()

visualizer = Visualizer(opt)


# create a webpage that summarizes the all results
web_dir = os.path.join(opt.results_dir, opt.name,
                       '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir,
                    'Experiment = %s, Phase = %s, Epoch = %s' %
                    (opt.name, opt.phase, opt.which_epoch))

app = Flask(__name__)
GRAY_LABEL_MAP = {
                  187: 156, # sky
                  123: 161, # stone
                  77: 168, # tree
                  69: 182, # wood
                  30: 154, # sea
                  161: 149,# rock
                  153: 123,# grass
                  217: 153,# sand
                  105: 0, # person
                  244: 158, # snow 
                  135: 178, # water
                  255: 182} # nothing

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print("Before process: ", set(gray.flatten().tolist()))
    mask = np.ones_like(gray) * 255
    for g_key, label in GRAY_LABEL_MAP.items():
        mask[gray == g_key] = label
    return mask

@app.route("/", methods=['GET'])
def hello():
    return "Hello !!!!!!"

@app.route("/generate", methods=['POST'])
def post():
    r = request.get_json()
    print("Response recieved.")
    raw_b64 = r['b64'].partition(",")[2]
    pad = len(raw_b64) % 4
    raw_b64 += "=" * pad
    decoded = b64decode(raw_b64.strip('base64'))
    arr = np.frombuffer(decoded, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    print(img.shape)
    img = preprocess(img)
    print(set(img.flatten().tolist()))
    # assert img.all() == 156
    cv2.imwrite("C:/python_projects/gaugan/SPADE/custom_data/val_inst/custom_test.png", img.astype(np.uint8))
    print("Is saved?: ", cv2.imwrite("C:/python_projects/gaugan/SPADE/custom_data/val_label/custom_test.png", img.astype(np.uint8)))
    generate_image()
    with open("./results/coco_pretrained/test_latest/images/synthesized_image/custom_test.png", "rb") as image_file:
        encoded_string = b64encode(image_file.read())
    return  encoded_string

# test
def generate_image():
    dataloader = data.create_dataloader(opt)
    for i, data_i in enumerate(dataloader):
        if i * opt.batchSize >= opt.how_many:
            break

        generated = model(data_i, mode='inference')

        img_path = data_i['path']
        for b in range(generated.shape[0]):
            print('process image... %s' % img_path[b])
            visuals = OrderedDict([('input_label', data_i['label'][b]),
                                   ('synthesized_image', generated[b])])
            visualizer.save_images(webpage, visuals, img_path[b:b + 1])
    # webpage.save()

if __name__ == "__main__":
    subprocess.Popen([sys.executable, './keamind/manage.py', 'runserver'])
    subprocess.Popen([chrome_path, '--disable-web-security', '--disable-gpu', '--user-data-dir=~/chromeTemp.', "http://localhost:8000/gaugan"])
    app.run(debug=debug, host="127.0.0.1", port=80, use_reloader=False)
