## Nvidia GauGAN Graphical User Interface with Drawingboard.js

This repository is similar to demo of Nvidia's GauGAN, called [SPADE](https://github.com/NVlabs/SPADE), but using Drawingboard.js, running a Flask app on port 80 for generating image, and django server that hosts the Drawingboard for generating images. I created this way before Nvidia published their own demo just to test. And since they are down most of the time I decided to publish.

Draw basic images and click generate to generate photo realistic scenery images.

![Demo](https://thumbs.gfycat.com/BetterSociableKusimanse-size_restricted.gif)

### Requirements
- Python 3
- GPU
- Chrome (if you want to run it locally)

### Installation

1- Install PyTorch( >=1.1.0 ) and TorchVision ( >= 0.3.0 ) from their website with GPU support. 

If you already have CUDA 8 and appropriate CuDNN installed, you can continue

```pip3 install -r requirements.txt```

3- Go to Nvidia's github repo [SPADE](https://github.com/NVlabs/SPADE) and follow their directives and install it to a folder. Download the weight files too!

### Usage

``` python3 spade_app.py -s PATH/TO/SPADE/LOCAL/REPO/DIR -c PATH/TO/CHROME/BINARY ```

This will open a Chrome page on localhost:8000 with GUI after sometime.
Colors represent a class on COCO images. Darker green are `forests`, dark blue is `Sea` etc..

![Colours-image](https://i.ibb.co/1zsd30X/Screenshot-2.png)


## Known Problems

 - Try to use fill colour as much as you can after selecting boundaries. This is about Drawingboard.js's inability to turn off anti-aliasing and creating non-solid colors all over with the Pen tool.
 - If fill colour fills entire board, you probably gone over the frame too fast with pen and there is gap.