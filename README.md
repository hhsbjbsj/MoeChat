# 基于GPT-SoVITS的语音交互系统
## 简介
一个非常基础的语音交互系统，使用GPT-SoVITS作为TTS模块，集成ASR接口，使用funasr作为语音识别模块基础。支持openai规范的大模型接口。
首Token延迟基本能做到1.5s以内。
### 测试平台
服务端
- OS: Manjaro
- CPU：R9 5950X
- GPU：RTX 3080ti

客户端
- 树莓派5

### 测试结果
![](screen/img.png)
## 整合包使用说明
### Windows
```bash
runtime\python.exe chat_server_Ver-0.1.py
```
### Linux
```bash
# 安装依赖
pip install -r extra-req.txt
pip install -r requirements.txt

# 运行
python chat_server_Ver-0.1.py
```
### 配置说明
整合包配置文件为config.yaml
```yaml
LLM:
  api: ""    # 大模型API
  key: ""    # 大模型API_Key
  model: ""  # 模型名称
GSV:
  text_lang: "zh"    # 合成文本的语种
  GPT_weight: ""     # GPT_weight模型路径
  SoVITS_weight: ""  # SoVITS_weight模型路径
  ref_audio_path: "" # 主要参考音频路径
  prompt_text: ""    # 参考音频文本
  prompt_lang: "zh"  # 参考音频语种
  aux_ref_audio_paths:  # 多参考音频
    -                # 多参考音频文件路径
  seed: -1           # 种子
```

### 简易客户端使用方法
## Windows
测试使用python 3.10
首先修改client.py文件asr_api、chat_api的ip地址。
```bash
# 创建python虚拟环境
python -m venv pp
.\pp\Scripts\pip.exe install -r client-requirements.txt

# 运行
.\pp\Scripts\python.exe client.py
```
