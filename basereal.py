###############################################################################
#  Copyright (C) 2024 LiveTalking@lipku https://github.com/lipku/LiveTalking
#  email: lipku@foxmail.com
# 
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
###############################################################################

import math
import torch
import numpy as np

import subprocess
import os
import time
import cv2
import glob
import resampy
import requests

import queue
from queue import Queue
from threading import Thread, Event
from io import BytesIO
import soundfile as sf

import av
from fractions import Fraction

from ttsreal import EdgeTTS,SovitsTTS,XTTS,CosyVoiceTTS,FishTTS,TencentTTS
from logger import logger

from tqdm import tqdm

# 后端API配置
BACKEND_API = "http://localhost:8888"  # 后端API地址

def read_imgs(img_list):
    frames = []
    logger.info('reading images...')
    for img_path in tqdm(img_list):
        frame = cv2.imread(img_path)
        frames.append(frame)
    return frames

class BaseReal:
    def __init__(self, opt):
        self.opt = opt
        self.sample_rate = 16000
        self.chunk = self.sample_rate // opt.fps # 320 samples per chunk (20ms * 16000 / 1000)
        self.sessionid = self.opt.sessionid
        self.username = self.opt.username if hasattr(self.opt, 'username') else 'default'
        self.backend_token = self.opt.backend_token if hasattr(self.opt, 'backend_token') else None

        # 从后端获取会话信息
        self._init_session_from_backend()

        if opt.tts == "edgetts":
            self.tts = EdgeTTS(opt,self)
        elif opt.tts == "gpt-sovits":
            self.tts = SovitsTTS(opt,self)
        elif opt.tts == "xtts":
            self.tts = XTTS(opt,self)
        elif opt.tts == "cosyvoice":
            self.tts = CosyVoiceTTS(opt,self)
        elif opt.tts == "fishtts":
            self.tts = FishTTS(opt,self)
        elif opt.tts == "tencent":
            self.tts = TencentTTS(opt,self)
        
        self.speaking = False

        self.recording = False
        self._record_video_pipe = None
        self._record_audio_pipe = None
        self.width = self.height = 0

        self.curr_state=0
        self.custom_img_cycle = {}
        self.custom_audio_cycle = {}
        self.custom_audio_index = {}
        self.custom_index = {}
        self.custom_opt = {}
        self.__loadcustom()

    def _init_session_from_backend(self):
        """从后端获取会话信息"""
        try:
            if not self.backend_token:
                logger.warning("No backend token provided, using default session")
                self.video_dir = os.path.join('videos', self.username)
                os.makedirs(self.video_dir, exist_ok=True)
                return

            # 创建会话
            response = requests.post(
                f"{BACKEND_API}/digital-person/create-session",
                headers={"Authorization": f"Bearer {self.backend_token}"}
            )
            
            if response.status_code == 200:
                data = response.json()
                self.sessionid = data["session_id"]
                # 确保视频目录存在
                self.video_dir = os.path.join('videos', self.username, str(self.sessionid))
                os.makedirs(self.video_dir, exist_ok=True)
                logger.info(f"Session initialized from backend: {self.sessionid}, video_dir: {self.video_dir}")
            else:
                logger.error(f"Failed to create session: {response.text}")
                raise Exception("Failed to create session from backend")
                
        except Exception as e:
            logger.error(f"Error initializing session from backend: {str(e)}")
            # 使用默认配置
            self.video_dir = os.path.join('videos', self.username, str(self.sessionid))
            os.makedirs(self.video_dir, exist_ok=True)

    def put_msg_txt(self,msg,eventpoint=None):
        self.tts.put_msg_txt(msg,eventpoint)
    
    def put_audio_frame(self,audio_chunk,eventpoint=None): #16khz 20ms pcm
        self.asr.put_audio_frame(audio_chunk,eventpoint)

    def put_audio_file(self,filebyte): 
        input_stream = BytesIO(filebyte)
        stream = self.__create_bytes_stream(input_stream)
        streamlen = stream.shape[0]
        idx=0
        while streamlen >= self.chunk:  #and self.state==State.RUNNING
            self.put_audio_frame(stream[idx:idx+self.chunk])
            streamlen -= self.chunk
            idx += self.chunk
    
    def __create_bytes_stream(self,byte_stream):
        #byte_stream=BytesIO(buffer)
        stream, sample_rate = sf.read(byte_stream) # [T*sample_rate,] float64
        logger.info(f'[INFO]put audio stream {sample_rate}: {stream.shape}')
        stream = stream.astype(np.float32)

        if stream.ndim > 1:
            logger.info(f'[WARN] audio has {stream.shape[1]} channels, only use the first.')
            stream = stream[:, 0]
    
        if sample_rate != self.sample_rate and stream.shape[0]>0:
            logger.info(f'[WARN] audio sample rate is {sample_rate}, resampling into {self.sample_rate}.')
            stream = resampy.resample(x=stream, sr_orig=sample_rate, sr_new=self.sample_rate)

        return stream

    def flush_talk(self):
        self.tts.flush_talk()
        self.asr.flush_talk()

    def is_speaking(self)->bool:
        return self.speaking
    
    def __loadcustom(self):
        for item in self.opt.customopt:
            logger.info(item)
            input_img_list = glob.glob(os.path.join(item['imgpath'], '*.[jpJP][pnPN]*[gG]'))
            input_img_list = sorted(input_img_list, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            self.custom_img_cycle[item['audiotype']] = read_imgs(input_img_list)
            self.custom_audio_cycle[item['audiotype']], sample_rate = sf.read(item['audiopath'], dtype='float32')
            self.custom_audio_index[item['audiotype']] = 0
            self.custom_index[item['audiotype']] = 0
            self.custom_opt[item['audiotype']] = item

    def init_customindex(self):
        self.curr_state=0
        for key in self.custom_audio_index:
            self.custom_audio_index[key]=0
        for key in self.custom_index:
            self.custom_index[key]=0

    def notify(self,eventpoint):
        logger.info("notify:%s",eventpoint)

    def start_recording(self):
        """开始录制视频"""
        if self.recording:
            return

        # 确保视频目录存在
        os.makedirs(self.video_dir, exist_ok=True)

        # 生成唯一的文件名
        timestamp = int(time.time())
        self.current_video_name = f"{self.sessionid}_{timestamp}.mp4"
        self.current_audio_name = f"{self.sessionid}_{timestamp}.aac"
        self.current_video_path = os.path.join(self.video_dir, self.current_video_name)
        self.current_audio_path = os.path.join(self.video_dir, self.current_audio_name)

        # 记录日志
        logger.info(f"Starting recording: video={self.current_video_path}, audio={self.current_audio_path}")

        command = ['ffmpeg',
                    '-y', '-an',
                    '-f', 'rawvideo',
                    '-vcodec','rawvideo',
                    '-pix_fmt', 'bgr24', #像素格式
                    '-s', "{}x{}".format(self.width, self.height),
                    '-r', str(25),
                    '-i', '-',
                    '-pix_fmt', 'yuv420p', 
                    '-vcodec', "h264",
                    self.current_video_path]
        self._record_video_pipe = subprocess.Popen(command, shell=False, stdin=subprocess.PIPE)

        acommand = ['ffmpeg',
                    '-y', '-vn',
                    '-f', 's16le',
                    '-ac', '1',
                    '-ar', '16000',
                    '-i', '-',
                    '-acodec', 'aac',
                    self.current_audio_path]
        self._record_audio_pipe = subprocess.Popen(acommand, shell=False, stdin=subprocess.PIPE)

        self.recording = True
        logger.info("Recording started")
    
    def record_video_data(self,image):
        if self.width == 0:
            print("image.shape:",image.shape)
            self.height,self.width,_ = image.shape
        if self.recording:
            self._record_video_pipe.stdin.write(image.tostring())

    def record_audio_data(self,frame):
        if self.recording:
            self._record_audio_pipe.stdin.write(frame.tostring())
		
    def stop_recording(self):
        """停止录制视频"""
        if not self.recording:
            return
            
        logger.info("Stopping recording...")
        self.recording = False 
        
        # 关闭视频管道
        if self._record_video_pipe and self._record_video_pipe.stdin:
            self._record_video_pipe.stdin.close()
        self._record_video_pipe.wait()
            
        # 关闭音频管道
        if self._record_audio_pipe and self._record_audio_pipe.stdin:
            self._record_audio_pipe.stdin.close()
        self._record_audio_pipe.wait()

        # 合并音视频
        output_path = os.path.join(self.video_dir, f"{self.sessionid}_{int(time.time())}.mp4")
        logger.info(f"Merging audio and video to: {output_path}")
        
        cmd_combine_audio = f"ffmpeg -y -i {self.current_audio_path} -i {self.current_video_path} -c:v copy -c:a copy {output_path}"
        os.system(cmd_combine_audio) 

        # 清理临时文件
        try:
            if os.path.exists(self.current_video_path):
                os.remove(self.current_video_path)
            if os.path.exists(self.current_audio_path):
                os.remove(self.current_audio_path)
            logger.info("Temporary files cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

        # 通知后端视频录制完成
        try:
            if self.backend_token:
                response = requests.post(
                    f"{BACKEND_API}/digital-person/notify-video-complete",
                    headers={"Authorization": f"Bearer {self.backend_token}"},
                    json={
                        "session_id": self.sessionid,
                        "video_path": output_path
                    }
                )
                if response.status_code != 200:
                    logger.error(f"Failed to notify backend: {response.text}")
        except Exception as e:
            logger.error(f"Error notifying backend: {str(e)}")

    def mirror_index(self,size, index):
        #size = len(self.coord_list_cycle)
        turn = index // size
        res = index % size
        if turn % 2 == 0:
            return res
        else:
            return size - res - 1 
    
    def get_audio_stream(self,audiotype):
        idx = self.custom_audio_index[audiotype]
        stream = self.custom_audio_cycle[audiotype][idx:idx+self.chunk]
        self.custom_audio_index[audiotype] += self.chunk
        if self.custom_audio_index[audiotype]>=self.custom_audio_cycle[audiotype].shape[0]:
            self.curr_state = 1  #当前视频不循环播放，切换到静音状态
        return stream
    
    def set_custom_state(self,audiotype, reinit=True):
        print('set_custom_state:',audiotype)
        self.curr_state = audiotype
        if reinit:
            self.custom_audio_index[audiotype] = 0
            self.custom_index[audiotype] = 0
    
    # def process_custom(self,audiotype:int,idx:int):
    #     if self.curr_state!=audiotype: #从推理切到口播
    #         if idx in self.switch_pos:  #在卡点位置可以切换
    #             self.curr_state=audiotype
    #             self.custom_index=0
    #     else:
    #         self.custom_index+=1