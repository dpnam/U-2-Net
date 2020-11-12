# U^2-Net (U square net)

## Demo: [Open Google Colab Notebook](https://colab.research.google.com/drive/1PsrXEAkgs3f1STMRP9OqrxYPQSSTxhrU?usp=sharing)

## Paper: [U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/pdf/2005.09007.pdf)

[Xuebin Qin](https://webdocs.cs.ualberta.ca/~xuebin/), <br/>
[Zichen Zhang](https://webdocs.cs.ualberta.ca/~zichen2/), <br/>
[Chenyang Huang](https://chenyangh.com/), <br/>
[Masood Dehghan](https://sites.google.com/view/masooddehghan), <br/>
[Osmar R. Zaiane](http://webdocs.cs.ualberta.ca/~zaiane/) and <br/>
[Martin Jagersand](https://webdocs.cs.ualberta.ca/~jag/).

__Contact__: xuebin[at]ualberta[dot]ca

## Download the pre-trained model
[u2net.pth (176.3 MB)](https://drive.google.com/file/d/1tA1efWGkM1BxnMxZ_amFDyVpNp1LEMst/view?usp=sharing) or [u2netp.pth (4.7 MB)](https://drive.google.com/file/d/18_q7KmanC25_zgCm9Pwsd4LHpQMCYe5B/view?usp=sharing) **(`u2netp.pth`available in `./weights/`)** and put it into the dirctory `./weights/`

## Required libraries

Python 3.6  
numpy 1.15.2  
scikit-image 0.14.0  
PIL 5.2.0  
PyTorch 0.4.0  
torchvision 0.2.1  
glob  
Tensorflow 2

## Create folder
- folder `images`: contain original images
- folder `videos`: contain original videos
- folder `results`: contain results both images and videos (remove background)

## Usage
- Image: `python u2net_image.py` / `python u2net_image.py u2net` / `python u2net_image.py u2netp`
- Video: `python u2net_video.py` / `python u2net_video.py u2net` / `python u2net_video.py u2netp`

**U-2-NET Paper:** [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007)

**Original Repo:** [U-2-Net Github repo](https://github.com/NathanUA/U-2-Net)

**References:** X. Qin, Z. Zhang, C. Huang, M. Dehghan, O. R. Zaiane, and M. Jagersand, “U2-net: Going deeper with nested u-structure for salient object
detection,” Pattern Recognition, vol. 106, p. 107404, 2020
