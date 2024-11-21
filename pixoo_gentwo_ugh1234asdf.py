import os
import cv2
import math
import random
import socket
import asyncio
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from tqdm import tqdm
import matplotlib as mpl
from time import sleep
from PIL import ImageOps
import matplotlib.pyplot as plt
from bleak import BleakScanner
from PIL import Image,ImageFile
from math import log10, ceil
from scipy.cluster import hierarchy
from random import randint
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image,ImageFile,ImageOps,ImageFont, ImageDraw
# import os
# import csv
# import cv2
# import math
# import torch
# import os,sys
# import random
# import shutil
# import socket
# import asyncio
# import hashlib
# import idx2numpy
# import numpy as np
# import pandas as pd
# import seaborn as sns
# from PIL import Image
# from tqdm import tqdm
# import matplotlib as mpl
# from time import sleep
# from PIL import ImageOps
# from sklearn import datasets
# from datetime import datetime
# import matplotlib.pyplot as plt
# from bleak import BleakScanner
# from numpy import linalg as LA
# from PIL import GifImagePlugin
# from PIL import Image,ImageFile
# from sklearn.utils import Bunch
# from dataclasses import dataclass
# from sklearn import preprocessing
# from math import log10, ceil, floor
# from scipy.cluster import hierarchy
# import matplotlib.animation as animation
# from random import randint,choices,choice
# from sklearn.preprocessing import normalize
# from sklearn.metrics import silhouette_score
# from matplotlib.animation import FuncAnimation
# from matplotlib.collections import LineCollection
# from random import randint,choices,choice,randrange
# from sklearn.cluster import AgglomerativeClustering
# from scipy.spatial.distance import pdist, squareform
# from traceback_with_variables import activate_by_import
# from pynvraw import api, NvError, get_phys_gpu, nvapi_api
# from random import randrange,randint,choice,choices,gauss
# from matplotlib.colors import BoundaryNorm, ListedColormap
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from PIL import Image,ImageFile,ImageOps,ImageFilter,ImageDraw,ImageDraw2
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from PIL import Image,ImageFile,ImageOps,ImageFilter,ImageChops,ImageFont, ImageDraw,ImageDraw2,PngImagePlugin


ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

img_types = [
            'blp', 'bmp', 'dib', 'bufr', 'cur'
            , 'pcx', 'dcx', 'dds', 'ps', 'eps'
            , 'fit', 'fits', 'fli', 'flc', 'ftc'
            , 'ftu', 'gbr', 'gif', 'grib', 'h5'
            , 'hdf', 'png', 'apng', 'jp2', 'j2k'
            , 'jpc', 'jpf', 'jpx', 'j2c', 'icns'
            , 'ico', 'im', 'iim', 'tif', 'tiff'
            , 'jfif', 'jpe', 'jpg', 'jpeg', 'mpg'
            , 'mpeg', 'mpo', 'msp', 'palm', 'pcd'
            , 'pxr', 'pbm', 'pgm', 'ppm', 'pnm'
            , 'psd', 'bw', 'rgb', 'rgba', 'sgi'
            , 'ras', 'tga', 'icb', 'vda', 'vst'
            , 'webp', 'wmf', 'emf', 'xbm', 'xpm'
            ,'nef'
            ]

RGB_PALETTE = {
        "3c":np.array([[255,255,255],[128,128,128],[0,0,0]]),
        "html":np.array([[255,160,122],[250,128,114],[233,150,122],[240,128,128],[205,92,92],[220,20,60],[178,34,34],[255,0,0],[139,0,0],[255,127,80],[255,99,71],[255,69,0],[255,215,0],[255,165,0],[255,140,0],[255,255,224],[255,250,205],[250,250,210],[255,239,213],[255,228,181],[255,218,185],[238,232,170],[240,230,140],[189,183,107],[255,255,0],[124,252,0],[127,255,0],[50,205,50],[0,255,0],[34,139,34],[0,128,0],[0,100,0],[173,255,47],[154,205,50],[0,255,127],[0,250,154],[144,238,144],[152,251,152],[143,188,143],[60,179,113],[46,139,87],[128,128,0],[85,107,47],[107,142,35],[224,255,255],[0,255,255],[0,255,255],[127,255,212],[102,205,170],[175,238,238],[64,224,208],[72,209,204],[0,206,209],[32,178,170],[95,158,160],[0,139,139],[0,128,128],[176,224,230],[173,216,230],[135,206,250],[135,206,235],[0,191,255],[176,196,222],[30,144,255],[100,149,237],[70,130,180],[65,105,225],[0,0,255],[0,0,205],[0,0,139],[0,0,128],[25,25,112],[123,104,238],[106,90,205],[72,61,139],[230,230,250],[216,191,216],[221,160,221],[238,130,238],[218,112,214],[255,0,255],[255,0,255],[186,85,211],[147,112,219],[138,43,226],[148,0,211],[153,50,204],[139,0,139],[128,0,128],[75,0,130],[255,192,203],[255,182,193],[255,105,180],[255,20,147],[219,112,147],[199,21,133],[255,255,255],[255,250,250],[240,255,240],[245,255,250],[240,255,255],[240,248,255],[248,248,255],[245,245,245],[255,245,238],[245,245,220],[253,245,230],[255,250,240],[255,255,240],[250,235,215],[250,240,230],[255,240,245],[255,228,225],[220,220,220],[211,211,211],[192,192,192],[169,169,169],[128,128,128],[105,105,105],[119,136,153],[112,128,144],[47,79,79],[0,0,0],[255,248,220],[255,235,205],[255,228,196],[255,222,173],[245,222,179],[222,184,135],[210,180,140],[188,143,143],[244,164,96],[218,165,32],[205,133,63],[210,105,30],[139,69,19],[160,82,45],[165,42,42],[128,0,0],[0,0,0]]),
        "basic":np.array([[0,0,0],[255,255,255],[255,0,0],[0,255,0],[0,0,255],[255,255,0],[0,255,255],[255,0,255],[192,192,192],[128,128,128],[128,0,0],[128,128,0],[0,128,0],[128,0,128],[0,128,128],[0,0,128],[0,0,0]]),
        "red":np.array([[255,160,122],[250,128,114],[233,150,122],[240,128,128],[205,92,92],[220,20,60],[178,34,34],[255,0,0],[139,0,0],[0,0,0]]),
        "orange":np.array([[255,127,80],[255,99,71],[255,69,0],[255,215,0],[255,165,0],[255,140,0],[0,0,0]]),
        "yellow":np.array([[255,255,224],[255,250,205],[250,250,210],[255,239,213],[255,228,181],[255,218,185],[238,232,170],[240,230,140],[189,183,107],[255,255,0],[0,0,0]]),
        "green":np.array([[124,252,0],[127,255,0],[50,205,50],[0,255,0],[34,139,34],[0,128,0],[0,100,0],[173,255,47],[154,205,50],[0,255,127],[0,250,154],[144,238,144],[152,251,152],[143,188,143],[60,179,113],[46,139,87],[128,128,0],[85,107,47],[107,142,35],[0,0,0]]),
        "teal":np.array([[224,255,255],[0,255,255],[0,255,255],[127,255,212],[102,205,170],[175,238,238],[64,224,208],[72,209,204],[0,206,209],[32,178,170],[95,158,160],[0,139,139],[0,128,128],[0,0,0]]),
        "blue":np.array([[176,224,230],[173,216,230],[135,206,250],[135,206,235],[0,191,255],[176,196,222],[30,144,255],[100,149,237],[70,130,180],[65,105,225],[0,0,255],[0,0,205],[0,0,139],[0,0,128],[25,25,112],[123,104,238],[106,90,205],[72,61,139],[0,0,0]]),
        "purple":np.array([[230,230,250],[216,191,216],[221,160,221],[238,130,238],[218,112,214],[255,0,255],[255,0,255],[186,85,211],[147,112,219],[138,43,226],[148,0,211],[153,50,204],[139,0,139],[128,0,128],[75,0,130],[0,0,0]]),
        "pink":np.array([[255,192,203],[255,182,193],[255,105,180],[255,20,147],[219,112,147],[199,21,133],[0,0,0]]),
        "white":np.array([[255,255,255],[255,250,250],[240,255,240],[245,255,250],[240,255,255],[240,248,255],[248,248,255],[245,245,245],[255,245,238],[245,245,220],[253,245,230],[255,250,240],[255,255,240],[250,235,215],[250,240,230],[255,240,245],[255,228,225],[0,0,0]]),
        "gray":np.array([[220,220,220],[211,211,211],[192,192,192],[169,169,169],[128,128,128],[105,105,105],[119,136,153],[112,128,144],[47,79,79],[0,0,0]]),
        "brown":np.array([[255,248,220],[255,235,205],[255,228,196],[255,222,173],[245,222,179],[222,184,135],[210,180,140],[188,143,143],[244,164,96],[218,165,32],[205,133,63],[210,105,30],[139,69,19],[160,82,45],[165,42,42],[128,0,0],[0,0,0]]),
        "GOLD":np.array([[250,250,210],[238,232,170],[240,230,140],[218,165,32],[255,215,0],[255,165,0],[255,140,0],[205,133,63],[210,105,30],[139,69,19],[160,82,45],[255,223,0],[212,175,55],[207,181,59],[197,179,88],[230,190,138],[153,101,21],[0,0,0]])
}

def f_split(f: str) -> list[str]:
    return [
        f[: len(f) - (f[::-1].find("/")) :].lower(),
        f[len(f) - (f[::-1].find("/")) : (len(f)) - 1 - len(f[-(f[::-1].find(".")) :])],
        f[-(f[::-1].find(".")) :].lower(),]
    
def statbar(tot:int,desc:str):
    l_bar='{desc}: {percentage:3.0f}%|'
    r_bar='| {n_fmt}/{total_fmt} [elapsed: {elapsed} / Remaining: {remaining}] '
    bar = '{rate_fmt}{postfix}]'
    status_bar = tqdm(total=tot, desc=desc,bar_format=f'{l_bar}{bar}{r_bar}')
    return status_bar

def imcb(image):
    def cb(img:np.array,tol:int=80)->list:
        mask = img>tol
        if img.ndim==3:
            mask = np.array(mask).all(3)
        m,n = mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        cs,ce = mask0.argmax(),n-mask0[::-1].argmax()
        rs,re = mask1.argmax(),m-mask1[::-1].argmax()
        return [rs,re,cs,ce]
    imgrey = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    ci = cb(imgrey,tol=40)
    return image[ci[0]:ci[1],ci[2]:ci[3]]

def _pkimg(img:np.uint8)->Image.Image:
    return ImageOps.contain(Image.fromarray(img).convert(mode="P", palette=Image.ADAPTIVE, colors=256).convert(mode="RGBA"),(32,32),Image.LANCZOS)

def vid_extract(file:str)->list:
    frame_list:list=[]
    vidcap = cv2.VideoCapture(file)
    success,image = vidcap.read()
    count = 0
    if success: 
        while success:
            count = count + 1
            frame_list.append((_pkimg(image)))
            success,image = vidcap.read()
    return frame_list

basic_colors:list=[
                    (255,255,255),
                    (255,0,0),
                    (0,255,0),
                    (0,0,255),
                    (255,255,0),
                    (0,255,255),
                    (255,0,255),
                    (192,192,192),
                    (128,128,128),
]

def imcb(image):
    def cb(img:np.array,tol:int=40)->list:
        mask = img>tol
        if img.ndim==3:
            mask = np.array(mask).all(3)
        m,n = mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        cs,ce = mask0.argmax(),n-mask0[::-1].argmax()
        rs,re = mask1.argmax(),m-mask1[::-1].argmax()
        return [rs,re,cs,ce]
    imgrey = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    ci = cb(imgrey,tol=40)
    return image[ci[0]:ci[1],ci[2]:ci[3]]

def imgtxt(txt:str=""):
    random.shuffle([x for x in basic_colors])
    fnt = ImageFont.truetype("E:/rule34/JetBrainsMono-ExtraBold.ttf",randint(18,24))
    img = Image.fromarray(np.uint8(np.zeros((32,32,3))),mode="RGB")
    imdraw = ImageDraw.Draw(img)
    _fill = basic_colors[randint(0,len(basic_colors)-1)]
    imdraw.text((0,4),txt,font=fnt,align="center",fill=_fill)
    return np.uint8(img)[:,:,::-1]

def txtarray_v(txt:str="")->np.uint8:
    lorem_arr = np.array_split([l for l in txt],len(txt))
    lstack=np.uint8(np.zeros((32,32,3)))
    for l in lorem_arr:
        tstack = imcb(imgtxt(''.join(c for c in l)))
        tstack = np.uint8(ImageOps.contain(Image.fromarray(tstack[:,:,::-1],mode="RGB"),(28,28)))[:,:,::-1]
        zstack = np.uint8(np.zeros((tstack.shape[0],32,3)))
        zstack[0:,:tstack.shape[1]:1] = tstack[::1,::1]
        lstack = np.vstack([lstack,zstack])
    print(lstack.shape)
    lstack = np.vstack([lstack,np.uint8(np.zeros((32,32,3)))])
    return lstack

def scroll_text(text_str:str=f"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")->list:
    lstack = txtarray_v(text_str)
    cv2.imwrite("./testdraw.jpg",lstack)
    i_shp = 1
    h,w,p=lstack.shape
    stop:int=int(w)-33
    scroll_list:list=[]
    while i_shp != stop:
        if i_shp == stop:
            i_shp = 1
        pkimg = np.uint8(np.zeros((32,32,3)))
        pkimg[0:,::1] = np.uint8(lstack[:,i_shp:i_shp+32,:])
        cv2.imwrite("./testdraw.jpg",pkimg)
        img=Image.fromarray(pkimg).convert(mode="P", palette=Image.ADAPTIVE, colors=256).convert(mode="RGBA")
        scroll_list.append(img)
        i_shp = i_shp + 1
    return scroll_list

def img_prep(img:np.uint8):
    size = max(img.shape[0:2])
    pad_x = size - img.shape[1]
    pad_y = size - img.shape[0]
    pad_l = pad_x // 2
    pad_t = pad_y // 2
    img = np.pad(img, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=0)
    interp = cv2.INTER_LANCZOS4
    img = cv2.resize(img, (32,32), interpolation=interp)
    img = np.array(img).astype('uint8')[:, :, ::-1]
    return img

def imcb(image,tol:int=40):
    def cb(img:np.array,tol:int)->list:
        mask = img>tol
        if img.ndim==3:
            mask = np.array(mask).all(3)
        m,n = mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        cs,ce = mask0.argmax(),n-mask0[::-1].argmax()
        rs,re = mask1.argmax(),m-mask1[::-1].argmax()
        return [rs,re,cs,ce]
    imgrey = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    ci = cb(imgrey,tol)
    return image[ci[0]:ci[1],ci[2]:ci[3]]

def imgtxt(txt:str="",fntsize:int=12):
    random.shuffle([x for x in basic_colors])
    fnt = ImageFont.truetype("E:/rule34/JetBrainsMono-ExtraBold.ttf",fntsize)
    img = Image.fromarray(np.uint8(np.zeros((64,64,3))),mode="RGB")
    imdraw = ImageDraw.Draw(img)
    imdraw.text((0,4),txt,font=fnt,align="center",fill=(255,255,255),spacing=0.1)
    return ImageOps.contain(Image.fromarray(img_prep(imcb(np.uint8(img)[:,:,::-1],80)),mode="RGB"),(32,32))

def _pkimg(img:np.uint8)->Image.Image:
    return ImageOps.contain(Image.fromarray(img).convert(mode="P", palette=Image.ADAPTIVE, colors=256).convert(mode="RGBA"),(32,32),Image.LANCZOS)


class PixooMax:  # PixooMax class, derives from Pixoo but does not support animation yet.
    CMD_SET_SYSTEM_BRIGHTNESS = 0x74
    CMD_SPP_SET_USER_GIF = 0xB1
    CMD_DRAWING_ENCODE_PIC = 0x5B
    BOX_MODE_CLOCK = 0
    BOX_MODE_TEMP = 1
    BOX_MODE_COLOR = 2
    BOX_MODE_SPECIAL = 3

    instance = None

    def __init__(self, mac_address):
        self.mac_address = mac_address #Constructor
        self.btsock = None

    @staticmethod
    def get():
        if PixooMax.instance is None:
            PixooMax.instance = PixooMax(PixooMax.BDADDR)
            PixooMax.instance.connect()
        return PixooMax.instance

    def connect(self):
        print(f"Connecting to {self.mac_address}...")
        self.btsock = socket.socket(socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM)
        self.btsock.connect((self.mac_address, 1))
        sleep(1)  # mandatory to wait at least 1 second
        print("Connected.")

    def __spp_frame_checksum(self, args):
        return sum(args[1:]) & 0xFFFF # Compute frame checksum

    def __spp_frame_encode(self, cmd, args):
        payload_size = len(args) + 3 # Encode frame for given command and arguments (list).
        frame_header = [1, payload_size & 0xFF, (payload_size >> 8) & 0xFF, cmd] # create our header
        frame_buffer = frame_header + args # concatenate our args (byte array)
        cs = self.__spp_frame_checksum(frame_buffer) # compute checksum (first byte excluded)
        frame_suffix = [cs & 0xFF, (cs >> 8) & 0xFF, 2] # create our suffix (including checksum)
        return frame_buffer + frame_suffix # return output buffer

    def send(self, cmd, args, retry_count=math.inf):
        spp_frame = self.__spp_frame_encode(cmd, args) # Send data to SPP. Try to reconnect if the socket got closed.
        self.__send_with_retry_reconnect(bytes(spp_frame), retry_count)

    def __send_with_retry_reconnect(self, bytes_to_send, retry_count=5):
        while retry_count >= 0: # Send data with a retry in case of socket errors.
            try:
                if self.btsock is not None:
                    self.btsock.send(bytes_to_send)
                    return
                print(f"[!] Socket is closed. Reconnecting... ({retry_count} tries left)")
                retry_count -= 1
                self.connect()
            except (ConnectionResetError, OSError):  # OSError is for Device is Offline
                self.btsock = None  # reset the btsock
                print("[!] Connection was reset. Retrying...")

    def set_system_brightness(self, brightness):
        self.send(PixooMax.CMD_SET_SYSTEM_BRIGHTNESS, [brightness & 0xFF]) #Set system brightness.

    def set_box_mode(self, boxmode, visual=0, mode=0):
        self.send(0x45, [boxmode & 0xFF, visual & 0xFF, mode & 0xFF]) #Set box mode.

    def set_color(self, r, g, b):
        self.send(0x6F, [r & 0xFF, g & 0xFF, b & 0xFF]) # Set color. 

    def img_prep(self, img):
        size = max(img.shape[0:2])
        pad_x = size - img.shape[1]
        pad_y = size - img.shape[0]
        pad_l = pad_x // 2
        pad_t = pad_y // 2
        img = np.pad(img, ((pad_t, pad_y - pad_t), (pad_l, pad_x - pad_l), (0, 0)), mode='constant', constant_values=0)
        interp = cv2.INTER_LANCZOS4
        img = cv2.resize(img, (32,32), interpolation=interp)
        img = np.array(img).astype('uint8')[:, :, ::-1]
        return img

    def encode_raw_image(self, img):
        try:
            w, h = img.size       
            if w == h:
                if w > 32:
                    img = ImageOps.contain(img,(32,32))
                pixels = []
                palette = []
                for y in range(32):
                    for x in range(32):
                        pix = img.getpixel((x, y))
                        if len(pix) == 4:
                            r, g, b, a = pix
                        elif len(pix) == 3:
                            r, g, b = pix
                        if (r, g, b) not in palette:
                            palette.append((r, g, b))
                            idx = len(palette) - 1
                        else:
                            idx = palette.index((r, g, b))
                        pixels.append(idx)
                bitwidth = ceil(log10(len(palette)) / log10(2))
                nbytes = ceil((256 * bitwidth) / 8.0)
                encoded_pixels = [0] * nbytes
                encoded_pixels = []
                encoded_byte = ""
                pixel_string = ""
                pixel_idx = []
                for i in pixels:
                    encoded_byte = bin(i)[2:].rjust(bitwidth, "0") + encoded_byte
                    # print(i,end="")
                    pixel_string = pixel_string + str(i)
                    pixel_idx.append(i)
                while len(encoded_byte) >= 8:
                    encoded_pixels.append(encoded_byte[-8:])
                    encoded_byte = encoded_byte[:-8]
                padding = 8 - len(encoded_byte)
                encoded_pixels.append(encoded_byte.rjust(bitwidth, "0"))
                encoded_data = [int(c, 2) for c in encoded_pixels]
                encoded_palette = []
                for r, g, b in palette:
                    encoded_palette += [r, g, b]
                return (len(palette), encoded_palette, encoded_data)
        except Exception as e:
            print(e)
            return
        finally: pass

    def pack_img(self, img):  #Draw encoded picture.
        nb_colors, palette, pixel_data = self.encode_raw_image(img)
        frame_size = 8 + len(pixel_data) + len(palette)
        frame_header = [
            0xAA,
            frame_size & 0xFF,
            (frame_size >> 8) & 0xFF,
            0,
            0,
            3,
            nb_colors & 0xFF,
            (nb_colors >> 8) & 0xFF,
        ]
        frame = frame_header + palette + pixel_data
        prefix = [0x0, 0x0A, 0x0A, 0x04]
        self.send(0x44, prefix + frame)

    def draw_file(self, file):
        img = cv2.imread(file)
        img = self.img_prep(img)
    
    
             


BT_MAC_ADDR = str("")
async def get_BTADDR():
    global BT_MAC_ADDR
    devices = await BleakScanner.discover()
    for d in devices:
        if str(d).find('Pixoo-Max')!=-1:
            print(str(d))
            BT_MAC_ADDR = str(d)[:17]

loop = asyncio.get_event_loop()
asyncio.run_coroutine_threadsafe(get_BTADDR(), loop)
while BT_MAC_ADDR == str(""):
    await get_BTADDR()
    asyncio.sleep(1.)

def main(BT_MAC_ADDR:str)->object:
    mac_addr = BT_MAC_ADDR
    print(mac_addr)
    pixoo = PixooMax(mac_addr)
    pixoo.connect()
    pixoo.set_system_brightness(10)
    return pixoo

pixoo = main(BT_MAC_ADDR)

pixoo.set_system_brightness(99)


from scipy.spatial import cKDTree
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.animation as animation
fig = plt.figure(figsize=(1,1),facecolor='black',tight_layout=True,dpi=1200)
plt.xticks([])
plt.yticks([])
plt.axis("off")
x:np.array
nx:np.array
p:np.array
pn:np.array
pdn:np.array
ndx:np.array

def _pkimg(img:np.uint8)->Image.Image:
    return ImageOps.contain(Image.fromarray(imcb(img)).convert(mode="P", palette=Image.ADAPTIVE, colors=16).convert(mode="RGBA"),(32,32),Image.LANCZOS)

def mk_plt(x:np.array,nx:np.array,p:np.array,pn:np.array,pdn:np.array,ndx:np.array,)->plt:
    return plt.plot(pdn[::1]-ndx[::-1]*-np.tan(x)*np.sin(p),(np.sin((nx)[::-1]-(np.pi)))*(+np.sin((pn)[::1]-(np.sqrt(12)/2))))

LN:int = 0

def imcb(image):
    def cb(img:np.array,tol:int=80)->list:
        mask = img>tol
        if img.ndim==3:
            mask = np.array(mask).all(3)
        m,n = mask.shape
        mask0,mask1 = mask.any(0),mask.any(1)
        cs,ce = mask0.argmax(),n-mask0[::-1].argmax()
        rs,re = mask1.argmax(),m-mask1[::-1].argmax()
        return [rs,re,cs,ce]
    imgrey = np.array(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY))
    ci = cb(imgrey,tol=40)
    return image[ci[0]:ci[1]+1,ci[2]:ci[3]]

def anim(ln):
    fig.clear()
    plt.axis("off")
    fig.set_facecolor("black")
    fig.set_alpha(0.5)
    fig.set_dpi(288)
    fig.set_linewidth(0.5)
    fig.set_edgecolor("black")
    fig.add_gridspec(32, 32)
    fig.set_tight_layout(True)
    ln = int(abs(ln+2))
    x = np.linspace(12,-144,ln)
    nx=(np.empty(x.shape))
    p = np.linspace(-12,144,ln)
    pn=(np.empty(p.shape))
    nx[::-1]=np.sin(x)[::-1]/1 - np.sin(x)[::1]/1
    x[::1]=np.sin(nx)[::-1]/(np.sqrt(12)) - np.sin(nx)[::-1]/(np.sqrt(12))
    nx[::-1]=np.sin(nx)[::-1]/1 - np.sin(nx)[::1]*+1
    nx[::1]=np.sin(nx)[::-1]*(np.sqrt(12))/2 - np.sin(nx)[::1]*(np.sqrt(12))/2
    p[::1]=(p-np.min(p))/(np.max(p)-np.min(p))**(1--1)*+-1
    pn[::-1]=np.sin(p)[::-1]/1 - np.sin(p)[::1]/1
    p[::1]=np.sin(pn)[::-1]/np.pi - np.sin(pn)[::-1]/np.pi
    pn[::-1]=np.sin(pn)[::-1]/1 - np.sin(pn)[::1]*+1
    pn[::1]=np.sin(pn)[::-1]*np.pi/2 - np.sin(pn)[::1]*np.pi/2
    pdn=(pn-min(nx))/(max(pn)-min(pn))**(1--1)*+-1
    ndx=(nx-min(nx))/(max(nx)-min(nx))**(1--1)*+-1
    LN = ln
    _sb_.update(n=1)
    return mk_plt(x,nx,p,pn,pdn,ndx)


_fps = 50
_interval = 20
_frames = (_fps * _interval) // 2
_sb_ = statbar(_frames,"anim_gen")
anim = animation.FuncAnimation(fig, anim, frames = _frames, interval = _interval) 
anim.save(f'e:/pixoo/oscillatory_mechanics_test12.mp4', writer = 'ffmpeg', fps = _fps)
_sb_.close()
vidframes:list=[]
vidcap = cv2.VideoCapture('e:/pixoo/oscillatory_mechanics_test12.mp4')
success,image = vidcap.read()
count = 0
_sb_ = statbar(_frames,"anim_gen")

if success: 
    while success:
        try:
            count = count + 1
            image = imcb(image)
            vidframes.append(image.copy())
            _sb_.update(n=1)
        except Exception as e:
            print(e)
            pass
        success,image = vidcap.read()

_sb_.close()

vidframes2_pos:list=[]
vidframes2_neg:list=[]
k_list_pos:list=[]
k_list_neg:list=[]

def vframelistgen(image:Image,rgb_key:str,klist:list):
    image = Image.fromarray(imcb(image[:,:,::-1])).convert(mode="P", palette=Image.ADAPTIVE, colors=256).convert(mode="RGB")
    image = np.array(RGB_PALETTE[k][cKDTree(RGB_PALETTE[k]).query(image,k=1)[1]]).astype('uint8')
    image = ImageOps.contain(Image.fromarray(image),(32,32),Image.LANCZOS).convert('RGBA')
    klist.append(image)

Kbar = statbar(len(RGB_PALETTE.keys()), desc=r'RGB')

for k in RGB_PALETTE.keys():
    k_list_pos=[]
    k_list_neg=[]
    status_bar = statbar(len(vidframes), desc=r'POS')
    with ThreadPoolExecutor(16) as executor:
        futures = [
            executor.submit(vframelistgen,frame, str(k),k_list_pos) for frame in vidframes]
        for _ in as_completed(futures):
            status_bar.update(n=1)
    status_bar.close()
    status_bar = statbar(len(vidframes), desc=r'NEG')
    with ThreadPoolExecutor(16) as executor:
        futures = [
            executor.submit(vframelistgen,frame, str(k),k_list_neg) for frame in vidframes]
        for _ in as_completed(futures):
            status_bar.update(n=1)
    status_bar.close()
    vidframes2_pos.append(k_list_pos)
    vidframes2_neg.append(k_list_neg)
    Kbar.update(n=1)

Kbar.close()



while True:
    try:
        for x in vidframes2_pos[randint(0,len(vidframes2_pos)-1)]:
            pixoo.pack_img(
                _pkimg(np.uint8(x))
            )
            sleep(0.05)
        for x in vidframes2_neg[randint(0,len(vidframes2_neg)-1)]:
            pixoo.pack_img(_pkimg(np.uint8(x)))
            sleep(0.05)
    except KeyboardInterrupt:
        break
    except Exception as e:
        pass


