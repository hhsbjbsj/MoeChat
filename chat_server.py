# Versdion 0.99

# 2025.05.13
# 修复一些bug
# 新增功能：声纹识别、更具情绪标签选择指定参考音频

# 2025.06.11
# 增加角色模板功能，可以使用内置提示词模板创建角色
# 增加日记系统（长期记忆），使ai可以记住所有的聊天内容，
#   并且可以使用像”昨天聊了什么“、”上周去了哪里“和”今天中午吃了什么“这样的语句进行基于时间范围的精确查询，不会像传统向量数据库那样因为时间维度而丢失记忆
# 增加核心记忆功能，使ai可以记住关于用户的重要回忆、信息和个人喜好
# 上述功均需要启用角色模板功能
# 脱离原有的GPT-SoVITS代码，改为API接口调用


import yaml

# 读取配置文件
config_data = {}
with open("config.yaml", "r", encoding="utf-8") as file:
    config_data = yaml.safe_load(file)

# 设置配置文件
# api_type = str(config_data["LLM"]["type"])
llm_api = config_data["LLM"]["api"]
llm_key = config_data["LLM"]["key"]
model1 = config_data["LLM"]["model"]
extra_config = config_data["LLM"]["extra_config"]
is_ttt =  config_data["Core"]["tt"]
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
if extra_config:
    for c in extra_config:
        req_data[c] = extra_config[c]

import os
import sys
now_dir = os.getcwd()
sys.path.append(now_dir)
import requests
import json
import time
import asyncio
from threading import Thread


# 创建数据文件夹
os.path.exists("data") or os.mkdir("data")

import base64
# import numpy as np
# import soundfile as sf
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi import FastAPI
import uvicorn
from io import BytesIO
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
from utilss.sv import SV
from utilss.agent import Agent
# import torch
import re
# import unicodedata
import jionlp

t2s_weights = config_data["GSV"]["GPT_weight"]
vits_weights =  config_data["GSV"]["SoVITS_weight"]
is_agent = config_data["Agent"]["is_up"]
if is_agent:
    a_config = {
        "char": config_data["Agent"]["char"],
        "user": config_data["Agent"]["user"],
        "is_data_base": config_data["Agent"]["lore_books"],
        "data_base_thresholds": config_data["Agent"]["books_thresholds"],
        "data_base_depth": config_data["Agent"]["scan_depth"],
        "is_long_mem": config_data["Agent"]["long_memory"],
        "is_check_memorys": config_data["Agent"]["is_check_memorys"],
        "mem_thresholds": config_data["Agent"]["mem_thresholds"],
        "char_settings": config_data["Agent"]["char_settings"],
        "char_personalities": config_data ["Agent"]["char_personalities"],
        "mask": config_data["Agent"]["mask"],
        "message_example":  config_data["Agent"]["message_example"],
        "prompt": config_data["Agent"]["prompt"],
        "is_core_mem": config_data["Agent"]["is_core_mem"],
        "llm": {"api": llm_api, "key": llm_key, "model": model1}
    }
    agent = Agent(a_config)
try:
    model_dir = "./utilss/models/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        device="cuda:0",
    )
except:
    print("[提示]未安装ASR模型，开始自动安装ASR模型。")
    from modelscope import snapshot_download
    model_dir = snapshot_download(
        model_id="iic/SenseVoiceSmall",
        local_dir="./utilss/models/SenseVoiceSmall",
        revision="master"
    )
    model_dir = "./utilss/models/SenseVoiceSmall"
    asr_model = AutoModel(
        model=model_dir,
        disable_update=True,
        device="cuda:0",
    )

# 载入声纹识别模型
sv_pipeline = ""
if config_data["Core"]["sv"]["is_up"]:
    sv_pipeline = SV(config_data["Core"]["sv"])
    is_sv = True
else:
    is_sv = False

# 提交到大模型
def to_llm(msg: list, res_msg_list: list, full_msg: list):
    global model1
    global llm_api
    global headers
    global req_data
    global emotion_list
    global is_ttt
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
    biao_dian_2 = ["…", "~", "~", "。", "？", "！", "?", "!"]
    biao_dian_3 = ["…", "~", "~", "。", "？", "！", "?", "!",  ",",  "，"]

    res_msg = ""
    tmp_msg = ""
    j = True
    j2 = True
    ref_audio = ""
    ref_text = ""
    biao_tmp = biao_dian_3
    stat = True
    for line in response.iter_lines():
        if line:
            try:
                if j:
                    print(f"\n[大模型延迟]{time.time() - t_t}")
                    j = False
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data:"):
                    data_str = decoded_line[5:].strip()
                    if data_str:
                        msg_t = json.loads(data_str)["choices"][0]["delta"]["content"]
                        res_msg += msg_t
                        tmp_msg += msg_t
                res_msg = res_msg.replace("...", "…")
                tmp_msg = tmp_msg.replace("...", "…")
            except:
                err = line.decode("utf-8")
                print(f"[错误]：{err}")
                continue
            # if not tmp_msg:
            #     continue
            ress = ""
            for ii in range(len(tmp_msg)):
                if tmp_msg[ii] == "(" or tmp_msg[ii] == "（":
                    stat = False
                    continue
                if tmp_msg[ii] == ")" or tmp_msg[ii] == "）":
                    stat = True
                    continue
                if tmp_msg[ii] in biao_tmp:
                    # 提取文本中的情绪标签，并设置参考音频
                    emotion = get_emotion(tmp_msg)
                    if emotion:
                        if emotion in emotion_list:
                            ref_audio = emotion_list[emotion][0]
                            ref_text = emotion_list[emotion][1]
                    ress = tmp_msg[:ii+1]
                    ress = jionlp.remove_html_tag(ress)
                    ttt = ress
                    if j2:
                        for i in range(len(ress)):
                            if ress[i] == "\n" or ress[i] == " ":
                                try:
                                    ttt = ress[i+1:]
                                except:
                                    ttt = ""
                    else:
                        pass
                        # if len(re.sub(r'[$(（].*?[）)]', '', ttt)) < 6:
                        #     continue
                    if ttt and stat:
                        res_msg_list.append([ref_audio, ref_text, ttt])
                    # print(f"[合成文本]{ress}")
                    if j2:
                        # biao_tmp = biao_dian_2
                        j2 = False
                    try:
                        tmp_msg = tmp_msg[ii+1:]
                    except:
                        tmp_msg = ""
                    break


    if len(tmp_msg) > 0:
        if emotion:
            if emotion in emotion_list:
                ref_audio = emotion_list[emotion][0]
                ref_text = emotion_list[emotion][1]
        res_msg_list.append([ref_audio, ref_text, tmp_msg])

    # 返回完整上下文 
    res_msg = jionlp.remove_html_tag(res_msg)
    ttt = ""
    for i in range(len(res_msg)):
        if res_msg[i] == "\n" or res_msg[i] == " ":
            try:
                ttt = res_msg[i+1:]
            except:
                ttt = ""
    full_msg.append(ttt)
    print(full_msg)
    # print(res_msg_list)
    res_msg_list.append("DONE_DONE")

def tts(datas: dict):
    global config_data
    res = requests.post(config_data["GSV"]["api"], json=datas, timeout=5)
    if res.status_code == 200:
        return res.content
    else:
        print(f"[错误]tts语音合成失败！！！")
        print(datas)
        return None

def clear_text(msg: str):
    msg = re.sub(r'[$(（].*?[）)]', '', msg)
    msg = jionlp.remove_exception_char(msg)
    return msg
# TTS并写入队
def to_tts(tts_data: list):
    global top_k
    global batch_size
    # def is_punctuation(char):
    #     return unicodedata.category(char).startswith('P')
    msg = clear_text(tts_data[2])
    # print(f"[实际输入文本]{tts_data[2]}[tts文本]{msg}")
    if len(msg) == 0:
        return "None"
    ref_audio = tts_data[0]
    ref_text = tts_data[1]
    global config_data
    datas = {
        "text": msg,
        "text_lang": config_data["GSV"]["text_lang"],
        "ref_audio_path": config_data["GSV"]["ref_audio_path"],
        "prompt_text": config_data["GSV"]["prompt_text"],   
        "prompt_lang": config_data["GSV"]["prompt_lang"],
    }
    if config_data["GSV"]["ex_config"]:
        for key in config_data["GSV"]["ex_config"]:
            datas[key] = config_data["GSV"]["ex_config"][key]
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
        global is_agent
        if is_agent:
            global agent
        # print(params)
        res_list = []
        audio_list = []
        full_msg = []
        if is_agent:
            t = time.time()
            msg_list = agent.get_msg_data(params.msg[-1]["content"])
            print(f"[提示]获取上下文耗时：{time.time() - t}")
        else:
            msg_list = params.msg
        llm_t = Thread(target=to_llm, args=(msg_list, res_list, full_msg, ))
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
                    if is_agent:    # 刷新智能体上下文内容
                        agent.add_msg(re.sub(r'<.*?>', '', full_msg[0]).strip())
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
    # if len(t2s_weights) != 0:
    #     tts_pipeline.init_t2s_weights(t2s_weights)
    # if len(vits_weights) != 0:
    #     tts_pipeline.init_vits_weights(vits_weights)

    # import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
