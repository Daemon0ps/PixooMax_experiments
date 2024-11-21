import os
import cv2
import math
import random
import socket
import asyncio
import numpy as np
from PIL import Image
from tqdm import tqdm
import multiprocessing
from time import sleep
from PIL import ImageOps
from random import randint
from math import log10, ceil
from bleak import BleakScanner
import matplotlib.pyplot as plt
from PIL import Image,ImageFile
from PIL import Image,ImageFile,ImageOps,ImageFont, ImageDraw
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
    fnt = ImageFont.truetype("./JetBrainsMono-ExtraBold.ttf",fntsize)
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
    
def _pkimg(img:np.uint8)->Image.Image:
    return ImageOps.contain(Image.fromarray(imcb(img)).convert(mode="P", palette=Image.ADAPTIVE, colors=16).convert(mode="RGBA"),(32,32),Image.LANCZOS)

def mk_plt(x:np.array,nx:np.array,p:np.array,pn:np.array,pdn:np.array,ndx:np.array,)->plt:
    return plt.plot(pdn[::1]-ndx[::-1]*-np.tan(x)*np.sin(p),(np.sin((nx)[::-1]-(np.pi)))*(+np.sin((pn)[::1]-(np.sqrt(12)/2))))

def imgtxt(txt:str="",fntsize:int=12,_fill=(255,255,255))->np.uint8:
    fnt = ImageFont.truetype("./JetBrainsMono-ExtraBold.ttf",fntsize)
    img = Image.fromarray(np.uint8(np.zeros((64,64,3))),mode="RGB")
    imdraw = ImageDraw.Draw(img)
    imdraw.text((0,4),txt,font=fnt,align="center",spacing=0.1,fill=_fill)
    img = imcb(np.uint8(img),20)
    return img

def scroll_text(txt:str="",fntsize:int=12,fntclr=(255,32,32))->list:
    def st_txt(txt:str="",fntsize:int=12,fntclr=(255,32,32))->np.uint8:
        fnt = ImageFont.truetype(font="./JetBrainsMono-ExtraBold.ttf",size=fntsize)
        img = Image.fromarray(np.uint8(np.zeros((64,1786,3))),mode="RGB")
        imdraw = ImageDraw.Draw(img)
        imdraw.text((0,4),txt,font=fnt,align="center",spacing=0.1,fill=fntclr)
        img = imcb(np.uint8(img),20)
        return img
    im_hstack = st_txt(txt,fntsize,fntclr)
    im_hstack = cv2.resize(imcb(im_hstack,20),(im_hstack.shape[1],10),interpolation=cv2.INTER_LANCZOS4)
    txtbuf = np.uint8(np.zeros((10,32,3)))
    im_hstack = np.hstack([txtbuf,im_hstack,txtbuf])
    i_shp = 1
    h,w,p = im_hstack.shape
    stop:int=int(w)-33
    scroll_list:list=[]
    while i_shp != stop:
        if i_shp == stop:
            break
        pkimg = np.uint8(np.zeros((10,32,3)))
        pkimg = np.uint8(im_hstack[:,i_shp:i_shp+32,:])
        scroll_list.append(pkimg)
        i_shp = i_shp + 1
    return scroll_list
             

BT_MAC_ADDR = str("")

def pixoo_main()->PixooMax:
    async def run():
        global BT_MAC_ADDR
        devices = await BleakScanner.discover()
        for d in devices:
            if str(d).find('Pixoo-Max')!=-1:
                print(str(d))
                BT_MAC_ADDR = str(d)[:17]
    loop = asyncio.get_event_loop()
    loop.run_until_complete(run())
    mac_addr = BT_MAC_ADDR
    print(mac_addr)
    pixoo = PixooMax(mac_addr)
    pixoo.connect()
    pixoo.set_system_brightness(95)
    return pixoo

if __name__ == "__main__":
    multiprocessing.freeze_support()
    pixoo = pixoo_main()
    sl = scroll_text("eat_my_ass")
    for i,frame in enumerate(sl):
        frame = cv2.resize(np.uint8(frame),(32,10),interpolation=cv2.INTER_LANCZOS4)
        img = ImageOps.contain(Image.fromarray(img_prep(imcb(np.uint8(frame)[:,:,::-1],20)),mode="RGB"),(32,32))
        pixoo.pack_img(_pkimg(np.uint8(img)))
