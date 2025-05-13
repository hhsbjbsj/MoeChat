# Versdion 0.1.8

# 2025.05.13
# 修复一些bug
# 新增功能：声纹识别、更具情绪标签选择指定参考音频

import yaml

# 读取配置文件
config_data = {}
with open("config.yaml", "r", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)

llm_api = config_data["LLM"]["api"]
llm_key = config_data["LLM"]["key"]
model1 = config_data["LLM"]["model"]
extra_config = config_data["LLM"]["extra_config"]
emotion_list = {}
if "extra_ref_audio" in config_data:
    emotion_list = config_data["extra_ref_audio"]
batch_size = 1
top_k = 15
try:
    batch_size = int(config_data["GSV"]["batch_size"])
except:
    batch_size = 1
try:
    top_k = int(config_data["GSV"]["top_k"])
except:
    top_k = 15

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {llm_key}"
}

req_data = {
    "model": model1 ,
    "stream": True
}
for c in extra_config:
    req_data[c] = extra_config[c]

import os
import sys
import requests
import json
import time
import asyncio
from threading import Thread

now_dir = os.getcwd()
sys.path.append(now_dir)
sys.path.append("%s/GPT_SoVITS" % (now_dir))

import argparse
import base64
import numpy as np
import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI
import uvicorn
from io import BytesIO
# from tools.i18n.i18n import I18nAuto
from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config
from GPT_SoVITS.TTS_infer_pack.text_segmentation_method import get_method_names as get_cut_method_names
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from utilss.sv import SV
import torch
import re

# print(sys.path)
# i18n = I18nAuto()
cut_method_names = get_cut_method_names()

parser = argparse.ArgumentParser(description="GPT-SoVITS api")
parser.add_argument("-c", "--tts_config", type=str, default="GPT_SoVITS/configs/tts_infer.yaml", help="tts_infer路径")
parser.add_argument("-a", "--bind_addr", type=str, default="127.0.0.1", help="default: 127.0.0.1")
parser.add_argument("-p", "--port", type=int, default="9880", help="default: 9880")
args = parser.parse_args()
config_path = args.tts_config
# device = args.device
port = args.port
host = args.bind_addr
argv = sys.argv

if config_path in [None, ""]:
    config_path = "GPT-SoVITS/configs/tts_infer.yaml"

tts_config = TTS_Config(config_path)
t2s_weights = config_data["GSV"]["GPT_weight"]
vits_weights =  config_data["GSV"]["SoVITS_weight"]
if t2s_weights not in [None, ""]:
    tts_config.t2s_weights_path = t2s_weights
else:
    print("[提示]：未设置GPT模型。")
if vits_weights not in [None, ""]:
    tts_config.vits_weights_path = vits_weights
else:
    print("[提示]：未设置SoVITS模型。")
if torch.cuda.is_available():
    tts_config.device = "cuda"
    print("CUDA可用")
else:
    tts_config.device = "cpu"
    print("CUDA不可用")
tts_config.update_configs()

print(tts_config)
tts_pipeline = TTS(tts_config)

model_dir = "iic/SenseVoiceSmall"
asr_model = AutoModel(
    model=model_dir,
    disable_update=True,
    #trust_remote_code=True,
    # remote_code="./model.py",
    # vad_model="fsmn-vad",
    # vad_kwargs={"max_single_segment_time": 30000},
    device="cuda:0",
)

sv_pipeline = ""
if config_data["Core"]["sv"]["is_up"]:
    sv_pipeline = SV(config_data["Core"]["sv"])
    is_sv = True
else:
    is_sv = False
# vad_model = AutoModel(model="fsmn-vad", disable_update=True)


def pack_wav(io_buffer:BytesIO, data:np.ndarray, rate:int):
    io_buffer = BytesIO()
    sf.write(io_buffer, data, rate, format='wav')
    return io_buffer

def tts(datas):
    global config_data
    try:
        tts_generator=tts_pipeline.run(datas)
        sr, audio_data = next(tts_generator)

        # vad判断有效音频起始点 并切割音频
        # print(audio_data)
        # print(len(audio_data))
        # try:
        #     if config_data["Main"]["is_vad"]:
        #         res = vad_model.generate(input = audio_data)[0]["value"]
        #         print(f"\n\n[VAD结果]{res}\n\n")
        #         if res:
        #             if res[0][0] > -config_data["Main"]["offset"] or config_data["Main"]["offset"] >= 0:
        #                 audio_data = audio_data[int(res[0][0] * sr) + config_data["Main"]["offset"]:]
        #             else:
        #                 audio_data = audio_data[int(res[0][0] * sr):]
        # except:
            # printp("\n\nVAD错误\n\n")
        audio_data = pack_wav(BytesIO(), audio_data, sr).getvalue()
        return audio_data
    except:
        return
# 提交到大模型
def to_llm(msg: list, res_msg_list: list, full_msg: list):
    global model1
    global llm_api
    global headers
    global req_data
    global emotion_list

    def get_emotion(msg: str):
        res = re.findall(r'\[(.*?)\]', msg)
        if len(res) > 0:
            match = res[-1]
            if match and emotion_list:
                if match in emotion_list:
                    return match

    data = req_data.copy()
    data["messages"] = msg
    t_t = time.time()
    try:
        response = requests.post(url = llm_api, json=data, headers=headers,stream=True)
    except:
        print("无法链接到LLM服务器")
        return JSONResponse(status_code=400, content={"message": "无法链接到LLM服务器"})
    
    # 信息处理
    # biao_dian = ["，", "。", "？", "！", ",", ".", "...", "?", "!", "、", "~", "~", "…"]
    # biao_dian = ["，", "。", "？", "！", ",", ".", "...", "?", "!", "…"]
    biao_dian_2 = ["…", "~", "~", "。", "？", "！", "?", "!"]
    biao_dian_3 = ["…", "~", "~", "。", "？", "！", "?", "!",  ",",  "，"]

    res_msg = ""
    tmp_msg = ""
    ttmp_msg = ""
    tt1 = False
    tt2 = True
    tt4 = True
    tt3 = 0
    j = True
    ref_audio = ""
    ref_text = ""
    for line in response.iter_lines():
        if line:
            try:
                if j:
                    print(f"\n[大模型延迟]{time.time() - t_t}")
                    j = False
                res_msg += json.loads(line.decode("utf-8").replace("data:", ""))["choices"][0]["delta"]["content"]
                tmp_msg += json.loads(line.decode("utf-8").replace("data:", ""))["choices"][0]["delta"]["content"]
                res_msg = res_msg.replace("...", "…")
                tmp_msg = tmp_msg.replace("...", "…")
            except:
                err = line.decode("utf-8")
                print(f"[错误]：{err}")
                continue

            # 提取""内的内容
            for m in range(len(tmp_msg)):
                if tmp_msg[m] == "\"":
                    if tt3 == 0:
                        tt1 = True
                        tt3 = 1
                        continue
                    if tt3 == 1:
                        tt1 = False
                        tt3 = 0
                        continue
                # 去除()（）包裹的内容
                if tmp_msg[m] == "(" or tmp_msg[m] == "（":
                    tt2 = False
                    continue
                if tmp_msg[m] == ")" or tmp_msg[m] == "）":
                    tt2 = True
                    continue
                if tt1 and tt2:
                    ttmp_msg += str(tmp_msg[m])
            tmp_msg = ""

            # 提取文本中的情绪标签，并设置参考音频
            emotion = get_emotion(res_msg)
            if emotion:
                if emotion in emotion_list:
                    ref_audio = emotion_list[emotion][0]
                    ref_text = emotion_list[emotion][1]

            start = 0
            for m in range(len(ttmp_msg)):
                if tt4:
                    if ttmp_msg[m] in biao_dian_3 and m > 0:
                        res_msg_list.append([ref_audio, ref_text, ttmp_msg[start : m+1]])
                        print(f"\n[合成文本]{ttmp_msg[start : m+1]}")
                        start = m + 1
                        tt4 = False
                else:
                    if ttmp_msg[m] in biao_dian_2 and m > 4:
                        res_msg_list.append([ref_audio, ref_text, ttmp_msg[start : m+1]])
                        print(f"\n[合成文本]{ttmp_msg[start : m+1]}")
                        start = m + 1
            
            if len(ttmp_msg) != 0:
                ttmp_msg = ttmp_msg[start:]
    if len(ttmp_msg) > 1:
        res_msg_list.append([ref_audio, ref_text, ttmp_msg])
    
    # 返回完整上下文 
    full_msg.append(res_msg)
    print(full_msg)
    # print(res_msg_list)
    res_msg_list.append("DONE_DONE")


# TTS并写入队
def to_tts(tts_data: list):
    global top_k
    global batch_size
    msg = tts_data[2]
    ref_audio = tts_data[0]
    ref_text = tts_data[1]
    if msg in ["…", "~", "~", "。", "？", "！", "?", "!",  ",",  "，"]:
        return "None"
    global config_data
    datas = {
        "text": msg,
        "text_lang": config_data["GSV"]["text_lang"],
        "ref_audio_path": config_data["GSV"]["ref_audio_path"],
        "aux_ref_audio_paths": config_data["GSV"]["aux_ref_audio_paths"],
        "prompt_text": config_data["GSV"]["prompt_text"],            
        "prompt_lang": config_data["GSV"]["prompt_lang"],      
        "top_k": top_k,
        "top_p": 1,
        "temperature": 1,
        "text_split_method": "cut0",
        "batch_size": batch_size,
        "batch_threshold": 0.75,
        "split_bucket": True,
        "speed_factor": 1,
        "fragment_interval": 0.3,
        "seed": config_data["GSV"]["seed"],
        "media_type": "wav",
        "streaming_mode": False,
        "parallel_infer": True,
        "repetition_penalty": 1.35,
        "sample_steps": 32,
        "super_sampling": False
    }
    if ref_audio:
        datas["ref_audio_path"] = ref_audio
        datas["prompt_text"] = ref_text
    try:
        byte_data = tts(datas)
        audio_b64 = base64.urlsafe_b64encode(byte_data).decode("utf-8")
        return audio_b64
    except:
        return "None"

def ttts(res_list: list, audio_list: list):
    i = 0
    while True:
        if i < len(res_list):
            if res_list[i] == "DONE_DONE":
                audio_list.append("DONE_DONE")
                print(f"完成...")
                break
            audio_list.append(to_tts(res_list[i]))
            i += 1
        time.sleep(0.05)

# asr功能
def asr(audio_data: bytes):
    global asr_model
    global is_sv
    global sv_pipeline

    tt = time.time()
    if is_sv:
        if not sv_pipeline.check_speaker(audio_data):
            return "None"
    # with open(f"./tmp/{tt}.wav", "wb") as file:
    #     file.write(audio_data)
    audio_data = BytesIO(audio_data)
    res = asr_model.generate(
        input=audio_data,
        # input=f"{model.model_path}/example/zh.mp3",
        cache={},
        language="zh", # "zh", "en", "yue", "ja", "ko", "nospeech"
        use_itn=True,
        batch_size=200,
    )
    # print(f"{model.model_path}/example/zh.mp3",)
    text = rich_transcription_postprocess(res[0]["text"])
    # text = res[0]["text"]
    print()
    print(f"[{time.time() - tt}]{text}\n\n")
    return text


# -----------------------------------API接口部分----------------------------------------------------------

app = FastAPI()

class tts_data(BaseModel):
    msg: list

async def text_llm_tts(params: tts_data, start_time):
        # print(params)
        res_list = []
        audio_list = []
        full_msg = []
        llm_t = Thread(target=to_llm, args=(params.msg, res_list, full_msg, ))
        llm_t.daemon = True
        llm_t.start()
        tts_t = Thread(target=ttts, args=(res_list, audio_list, ))
        tts_t.daemon = True
        tts_t.start()

        i = 0
        stat = True
        while True:
            if i < len(audio_list):
                if audio_list[i] == "DONE_DONE":
                    data = {"file": None, "message": full_msg[0], "done": True}
                    yield f"data: {json.dumps(data)}\n\n"
                data = {"file": audio_list[i], "message": res_list[i][2], "done": False}
                # audio = str(audio_list[i])
                # yield str(data)
                if stat:
                    print(f"\n[服务端首句处理耗时]{time.time() - start_time}\n")
                    stat = False
                yield f"data: {json.dumps(data)}\n\n"
                i += 1
            await asyncio.sleep(0.05)                

@app.post("/api/chat")
async def tts_api(params: tts_data):
    return StreamingResponse(text_llm_tts(params, time.time()), media_type="text/event-stream")

# asr接口
class asr_data(BaseModel):
    data: str
@app.post("/api/asr")
async def asr_api(params: asr_data):
    audio_data = base64.urlsafe_b64decode(params.data.encode("utf-8"))
    text = asr(audio_data)
    return text

# -----------------------------------API接口部分----------------------------------------------------------


if __name__ == "__main__":
    # global config_data
    t2s_weights = config_data["GSV"]["GPT_weight"]
    vits_weights =  config_data["GSV"]["SoVITS_weight"]
    if len(t2s_weights) != 0:
        tts_pipeline.init_t2s_weights(t2s_weights)
    if len(vits_weights) != 0:
        tts_pipeline.init_vits_weights(vits_weights)

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
