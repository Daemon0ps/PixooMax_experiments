import re
import cv2
import math
import codecs
import sys
import socket
import asyncio
import keyring
import paramiko
import numpy as np
from time import sleep
from random import randint
from typing import Optional
from math import ceil, log10
from datetime import datetime
from bleak import BleakScanner
from unidecode import unidecode
from dataclasses import dataclass
from paramiko.channel import ChannelFile
from paramiko import AutoAddPolicy, SSHClient, SFTPClient
from paramiko import BadHostKeyException,AuthenticationException,SSHException
from PIL import Image,ImageOps,ImageDraw,ImageFont
from requests import Response, ConnectionError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests_html import HTMLSession

IP_LIST = ['192.168.1.203','192.168.1.212']
BAN_LIST =  [
            'mojang.com ',
            'minecraftservices.com',
            'minecraft-services.net',
            'minecraft.net',
            'amazonvideo.com',
            'bing.com',
            'blogspot.com',
            'boredcomics.com',
            'boredpanda.com',
            'cursecdn.com',
            'curseforge.com',
            'deviantart.com',
            'deviantart.net',
            'discordapp.com',
            'disqus.com',
            'fandom.com',
            'fowllanguagecomics.com',
            'giphy.com',
            'github.io',
            'githubusercontent.com',
            'googlevideo.com',
            'hulu.com',
            'hulustream.com',
            'insightexpressai.com',
            'instagram.com',
            'netflix.com',
            'nflxext.com',
            'nflximg.net',
            'nflxso.net',
            'nflxvideo.net',
            'nianticlabs.com',
            'nintendo.com',
            'nintendo.net',
            'openai.com',
            'pikminwiki.com',
            'pinimg.com',
            'pinterest.com',
            'quidditchchampions.com',
            'rbx.com',
            'rbxcdn.com',
            'redd.it',
            'reddit.com',
            'redditmedia.com',
            'redditstatic.com',
            'reedpopcdn.com',
            'roblox.com',
            'scroll.com',
            'skype.com',
            'snapchat.com',
            'ssbwiki.com',
            'steamcontent.com',
            'steampowered.com',
            'steamserver.net',
            'steamstatic.com',
            'thegamer.com',
            'thegamerimages.com',
            'thepopverse.com',
            'thepositiveencourager.global',
            'truvidplayer.com',
            'tumblr.com',
            'twimg.com',
            'twitter.com',
            'unity3d.com',
            'youtube.com',
            'ytimg.com'
]

import ssl
import urllib.request
import urllib
from urllib.error import HTTPError
from time import sleep
from random import randint
from tqdm import tqdm

save_path = "./test/"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows; U; Win98; en-US; rv:1.6) Gecko/20040206 Firefox/0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Charset": "ISO-8859-1,utf-8;q=0.7,*;q=0.3",
    "Accept-Encoding": "none",
    "Accept-Language": "en-US,en;q=0.8",
    "Connection": "keep-alive",
}
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

def yb_img(i):
    try:
        sleep(randint(1, 5))
        req_url = "wttr.in/Beaverton?0pQAT"
        request = urllib.request.Request(req_url, None, headers=headers)
        response = urllib.request.urlopen(request, context=ctx)
        url_file = response.read()
        with open(save_path + str(i).zfill(4) + ".jpg", "wb") as fi:
            fi.write(url_file)
    except KeyboardInterrupt:
        sys.exit()
    except HTTPError as h_err:
        if h_err.find("404"):
            sys.exit()
    except Exception as e:
        print(e)
        pass

@dataclass(frozen=False)
class c:
    url0:str = str("")
    session = HTMLSession()
    retry = Retry(connect=3, backoff_factor=5)
    adapter = HTTPAdapter(max_retries=30)
    cxn_state:bool = False
    api: str = "http://127.0.0.1:7860"
    rr:Response = Response
    headers = {'User-Agent': "Mozilla/5.0 (Windows; U; Win98; en-US; rv:1.6) Gecko/20040206 Firefox/0.8",
                                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                                'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
                                'Accept-Encoding': 'none',
                                'Accept-Language': 'en-US,en;q=0.8',
                                'Connection': 'keep-alive'}
    def __post_init__(self):
        self.url0 = c.url0
        self.session = c.session
        self.retry = c.retry
        self.adapter = c.adapter
        self.cxn_state = c.cxn_state
        self.api = c.api
        self.rr = c.rr
        self.headers = c.headers
        super().__setattr__("attr_name", self)
    
    @staticmethod
    def _cxn()->HTMLSession():
        if c.cxn_state:
            try:
                c.session.close()
                c.cxn_state = False
            except Exception as cxe:
                pass
        c.session = HTMLSession()
        c.retry = Retry(connect=3, backoff_factor=5)
        c.adapter = HTTPAdapter(max_retries=30)
        c.session.mount('https://', c.adapter)
        c.cxn_state = True
        return c.session
        
            
    @staticmethod
    def _get(ep:str)->Response:
        resp:Response
        try:
            c.session = c._cxn()
            resp = c.session.get(ep, headers=c.headers)
            c.session.close()
            c.cxn_state = False
            return resp
        except KeyboardInterrupt:
            sys.exit()
        except ConnectionError as ce:
            print(ce)
            pass
        except Exception as e:
            print(e)
            pass

class CMD:
    f_stdout:ChannelFile
    stdout:str
    cmd_str:str
    def __init__(self,cmd_str)->None:
        self.hostname:str = keyring.get_password("pi_hole","ip_addr")
        self.port:int = keyring.get_password("pi_hole","port_num")
        self.username:str = keyring.get_password("pi_hole","user")
        self.password:str = keyring.get_password("pi_hole","password")
        self.stdout:str=""
        with SSHClient() as ssh_client:
            ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh_client.connect(hostname=self.hostname,port=self.port,username=self.username,password=self.password)
            self.cmd_str=cmd_str
            _, self.f_stdout, _ = ssh_client.exec_command(self.cmd_str, bufsize=- 1, timeout=None, get_pty=False, environment=None)
            self.stdout = unidecode(codecs.decode(self.f_stdout.read(),'utf-8'))

    def __str__(self):
        return f'{self.stdout}'


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
                    pixel_string = pixel_string + str(i)
                    pixel_idx.append(i)
                while len(encoded_byte) >= 8:
                    encoded_pixels.append(encoded_byte[-8:])
                    encoded_byte = encoded_byte[:-8]
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

def imcb(image,tol:int=40)->np.uint8:
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
    return np.uint8(image[ci[0]:ci[1],ci[2]:ci[3]])

def imgtxt(txt:str="",fntsize:int=12,_fill=(255,255,255))->np.uint8:
    fnt = ImageFont.truetype("/mnt/e/rule34/JetBrainsMono-ExtraBold.ttf",fntsize)
    img = Image.fromarray(np.uint8(np.zeros((64,64,3))),mode="RGB")
    imdraw = ImageDraw.Draw(img)
    imdraw.text((0,4),txt,font=fnt,align="center",spacing=0.1,fill=_fill)
    img = imcb(np.uint8(img),20)
    return img

def sshtxt(txt:str="",fntsize:int=12)->np.uint8:
    fnt = ImageFont.truetype("/mnt/e/rule34/JetBrainsMono-ExtraBold.ttf",fntsize)
    img = Image.fromarray(np.uint8(np.zeros((64,64,3))),mode="RGB")
    imdraw = ImageDraw.Draw(img)
    imdraw.text((0,4),txt,font=fnt,align="center",spacing=0.1,fill=(255,32,32))
    img = imcb(np.uint8(img),20)
    return img

def _pkimg(img:np.uint8)->Image.Image:
    return ImageOps.contain(Image.fromarray(img).convert(mode="P", palette=Image.ADAPTIVE, colors=256).convert(mode="RGBA"),(32,32),Image.LANCZOS)

def bancheck()->str:
    site_st:list=[]
    for ipaddr in IP_LIST:
        xs = CMD(cmd_str=f"tail -n 300 /var/log/pihole.log | grep '{ipaddr}'").__str__()
        xre = re.findall(r'(.*?) dnsmasq.*query.*[\]] (.*?) from (.*)\n',xs)
        _ = len(
                list(
                    map(
                        lambda x: site_st.append(re.findall(r'(.*?[.].*?)[.\n].*',str(x[1])[::-1])[0][::-1]) \
                                if x[1] not in site_st \
                                and len(re.findall(r'(.*?[.].*?)[.\n].*',str(x[1])[::-1])) > 0 \
                                else None,
                        [x for x in xre]
                        )
                    )
                )
        _
    unq_st = sorted(
        [x for x in np.unique(site_st) if str(BAN_LIST).lower().find(x)!=-1],key = lambda x: site_st.count(x), reverse = True)
    if len(unq_st)>0:
        return str(unq_st[0])[:unq_st[0].find('.'):]
    else:
        return "---"

def scroll_text(txt:str="",fntsize:int=12,fntclr=(255,32,32))->list:
    def st_txt(txt:str="",fntsize:int=12,fntclr=(255,32,32))->np.uint8:
        fnt = ImageFont.truetype(font="/mnt/z/JetBrainsMono-ExtraBold.ttf",size=fntsize)
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


def ret_dtime()->tuple:
    hrs = str(datetime.strftime(datetime.now(), r"%H"))
    mins = str(datetime.strftime(datetime.now(), r"%M"))
    img1 = imgtxt(f'{str(hrs).zfill(2)}{str(mins).zfill(2)}',24,(0,255,255))
    img1 = cv2.resize(imcb(img1,20),(32,10),interpolation=cv2.INTER_LANCZOS4)
    return img1

BT_MAC_ADDR = str("")
        
def pixoo_main():
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
    bc_bool:bool = False
    bc_counter = 0
    w_counter = 0
    sl:list = []
    loop_bool = True
    TODAY = f'{str(str(datetime.strftime(datetime.now(), r"%a")).upper())}.{str(int(str(datetime.strftime(datetime.now(), r"%m"))))}-{str(int(str(datetime.strftime(datetime.now(), r"%d"))))}'
    wttrin = f'https://wttr.in/Beaverton?u&0pQATF&format=%C+%t(+%f)+%w'
    wl:list = scroll_text(TODAY + chr(32) + str(c._get(wttrin).text).strip().upper().replace(chr(32)+chr(32),chr(32)),12,(255,255,32))
    wttr = len(wl)
    w_c = 0
    while loop_bool == True:
        sleep(0.05)
        TODAY = f'{str(str(datetime.strftime(datetime.now(), r"%a")).upper())}.{str(int(str(datetime.strftime(datetime.now(), r"%m"))))}-{str(int(str(datetime.strftime(datetime.now(), r"%d"))))}'
        bc_counter = 0
        b_c = bancheck()
        if str(int(str(datetime.strftime(datetime.now(), r"%H")))) in ['20','21','22','23','0','1','2','3','4','5','6']:
            pixoo.set_system_brightness(1)
            # pixoo.set_system_brightness(95)
        else: 
            pixoo.set_system_brightness(95)
            # pixoo.set_system_brightness(1)
        if b_c != "---":
            bc_bool = True
        else:
            bc_bool = False
        if w_c >= wttr:
            w_c = 0
        if w_counter >= 5000:
            wl:list = scroll_text(TODAY + chr(32) + str(c._get(wttrin).text).strip().replace(chr(34),'').upper().replace(chr(32)+chr(32),chr(32)),12,(255,255,32))
            wttr = len(wl)
        while bc_counter <= 200:
            if w_c >= wttr:
                w_c = 0
            img1 = ret_dtime()
            img = imcb(np.vstack([img1,np.uint8(np.zeros((1,32,3))),np.uint8(np.zeros((10,32,3))),np.uint8(np.zeros((1,32,3))),wl[w_c]]),40)
            img = ImageOps.contain(Image.fromarray(img_prep(imcb(np.uint8(img)[:,:,::-1],20)),mode="RGB"),(32,32))
            pixoo.pack_img(_pkimg(np.uint8(img)))
            sleep(0.1)
            while bc_bool == True:
                sl:list = scroll_text(b_c,12)
                for i,frame in enumerate(sl):
                    img1 = ret_dtime()
                    frame = cv2.resize(np.uint8(frame),(32,10),interpolation=cv2.INTER_LANCZOS4)
                    img = imcb(np.vstack([cv2.bitwise_not(img1),np.uint8(np.zeros((1,32,3))),frame,np.uint8(np.zeros((1,32,3))),wl[w_c]]),40)
                    img = ImageOps.contain(Image.fromarray(img_prep(imcb(np.uint8(img)[:,:,::-1],20)),mode="RGB"),(32,32))
                    pixoo.pack_img(_pkimg(np.uint8(img)))
                    w_c = w_c + 1
                    if w_c >= wttr:
                        w_c = 0
                    sleep(0.05)
                bc_bool = False
            bc_counter = bc_counter + 1
            w_counter = w_counter + 1
            w_c = w_c + 1


if __name__ == "__main__":
    pixoo_main()

