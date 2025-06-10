import sounddevice as sd
import soundfile as sf
from pysilero import VADIterator
import numpy as np
from scipy.signal import resample  # 导入重采样函数
import io
from threading import Thread
import time
import subprocess
import requests
import base64
import pygame
import json
import ui
import flet as ft

asr_api = "http://127.0.0.1:8001/api/asr"
chat_api = "http://127.0.0.1:8001/api/chat"
status = True

ai_name = "爱丽丝"
user_name = "测试用户"
system = ''''''

# 用于存储上下文内容
data = {
    "msg": [
        # {"role": "system", "content": system}
    ]
}

# # Chat messages
# ui.chat_list = ft.ListView(
#     controls=[],
#     expand=True,
#     spacing=10,
#     auto_scroll=True,
# )

def add_msg(str_msg: str):
    # global ui.chat_list
    global ai_name
    if ui.chat_list.controls[-1].data != ai_name:
        msg = ui.ChatMessage(ai_name, "", "left")
        msg.msg_list.controls.append(ui.get_msg_box(str_msg))
        ui.chat_list.controls.append(msg.cont)
    else:
        if ui.chat_list.controls[-1].alignment == ft.MainAxisAlignment.START:
            ui.chat_list.controls[-1].controls[-1].controls[-1].content.value += str_msg
        else:
            ui.chat_list.controls[-1].controls[0].controls[-1].content.value += str_msg
    ui.chat_list.update()
    ui.chat_list.scroll_to(offset=-1, duration=1000)

def add_msg_me(str_msg: str):
    # global ui.chat_list
    if len(ui.chat_list.controls) == 0:
        msg = ui.ChatMessage(user_name, "", "right")
        msg.msg_list.controls.append(ui.get_msg_box(str_msg))
        ui.chat_list.controls.append(msg.cont)
    else:
        if ui.chat_list.controls[-1].data != user_name:
            msg = ui.ChatMessage(user_name, "", "right")
            msg.msg_list.controls.append(ui.get_msg_box(str_msg))
            ui.chat_list.controls.append(msg.cont)
        else:
            if ui.chat_list.controls[-1].alignment == ft.MainAxisAlignment.START:
                ui.chat_list.controls[-1].controls[-1].controls.append(ui.get_msg_box(str_msg))
            else:
                ui.chat_list.controls[-1].controls[0].controls.append(ui.get_msg_box(str_msg))
    ui.chat_list.update()
    ui.chat_list.scroll_to(offset=-1, duration=1000)

def play_autio(audio_data, msg, t):
    if t != None:
        t.join()
    if audio_data == "None":
        print(msg)
        add_msg(msg)
        return
    print(msg)
    add_msg(msg)
    with open("tmp.wav", "wb") as file:
        file.write(audio_data)
    pygame.mixer.init()
    sound = pygame.mixer.Sound("tmp.wav")
    channel = pygame.mixer.Channel(0)  # 创建一个通道
    channel.play(sound)
    while channel.get_busy():
        time.sleep(0.1)

def to_llm_and_tts(msg: str, asr_c: str):
    global chat_api
    global data
    global status

    msg = msg.replace("\"", "")

    print(f"[ars耗时:{asr_c}]{msg}")
    status = False
    data["msg"].append(
        {
            "role": "user",
            "content": msg
        }
    )

    tt_t = time.time()
    #print(data)
    try:
        res = requests.post(chat_api, json=data, stream=True)
    except:
        print("\n[error]无法连接大模型服务器...")
        data["msg"].remove(-1)
        status = True
        print("\n")
        print("Me:")
        return
    tt = 1
    tmp_msg = ""
    tt_list = []
    for line in res.iter_lines():
        if line:
            # print(line.decode("utf-8")[6:])
            rec_data = json.loads(line.decode("utf-8")[6:])
            if rec_data["done"]:
                data["msg"].append(
                    {
                        "role": "assistant",
                        "content": rec_data["message"]
                    }
                )
                break
            audio_b64 = rec_data["file"]
            tmp = "None"
            if audio_b64 != "None":
                tmp = audio_b64.encode("utf-8")
                tmp = base64.urlsafe_b64decode(tmp)
            if tt:
                print(f"\n首token耗时:{(time.time() - tt_t):.3f}")
                print("Ai: ")
                tt = 0
            tmp_msg += rec_data["message"].replace("\n", "")
            if len(tt_list) == 0:
                t = Thread(target=play_autio, args=(tmp, rec_data["message"], None, ))
                t.daemon = True
                tt_list.append(t)
                t.start()
            else:
                t = Thread(target=play_autio, args=(tmp, rec_data["message"], tt_list[-1], ))
                t.daemon = True
                tt_list.append(t)
                t.start()

    for t in tt_list:
        t.join()
    time.sleep(0.3)
    status = True
    print("\n")
    print("Me:")

def to_asr(audio: bytes, t):
    global asr_api
    audio64 = base64.urlsafe_b64encode(audio).decode("utf-8")
    ddd = {"data": audio64}
    res = requests.post(asr_api, json=ddd)
    # asr耗时
    t = f"{(time.time() - t):.3f}"
    if res.text == "\"None\"":
        return
    add_msg_me(res.text.replace("\"", ""))
    # print(f"\nAi:")
    to_llm_and_tts(res.text, t)

def gen_audio(current_speech):
    t = time.time()
    combined = np.concatenate(current_speech)
    audio_bytes = b""
    with io.BytesIO() as buffer:
        sf.write(
            buffer,
            combined,
            16000,
            format="WAV",
            subtype="PCM_16",
        )
        buffer.seek(0)
        audio_bytes = buffer.read()  # 完整的 WAV bytes
    to_asr(audio_bytes, t)

def main():
    global status
    # 初始化 VAD 迭代器，指定采样率为 16000Hz
    vad_iterator = VADIterator(speech_pad_ms=100)

    # 查询音频设备
    devices = sd.query_devices()
    if len(devices) == 0:
        print("No microphone devices found")
    subprocess.run('clear', shell=True)
    print(devices)
    default_input_device_idx = sd.default.device[0]
    print(f'Use default device: {devices[default_input_device_idx]["name"]}')
    device_id = int(input("输入麦克风序号: "))
    # 计算每次读取的样本数（保持 48000Hz 的输入）
    samples_per_read = int(0.1 * 48000)  # 48000Hz * 0.1s = 4800 samples

    # 储存识别到的音频片段
    current_speech = []
    print("启动成功")
    print("Me:")
    with sd.InputStream(channels=1, dtype="float32", samplerate=48000, device=device_id) as s:
        while True:
            # 读取音频数据
            samples, _ = s.read(samples_per_read)

            # 转换为单声道并重采样到 16000Hz
            samples_mono = samples[:, 0].astype(np.float32)  # 确保为 float32
            resampled = resample(samples_mono, 1600)         # 4800 → 1600 样本
            resampled = resampled.astype(np.float32)         # 保持数据类型一致

            # 将重采样后的数据传递给 VAD 处理
            for speech_dict, speech_samples in vad_iterator(resampled):
                if "start" in speech_dict:
                    current_speech = []
                    pass
                if status:
                    current_speech.append(speech_samples)
                else:
                    continue
                is_last = "end" in speech_dict
                if is_last:
                    t = Thread(target=gen_audio, args=(current_speech.copy(), ))
                    t.daemon = True
                    t.start()
                    current_speech = []  # 清空当前段落
