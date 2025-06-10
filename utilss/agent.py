# 角色模板

import os
from utilss import long_mem, data_base, prompt, core_mem
import time
from threading import Thread
import requests
import jionlp
import ast
# from ruamel.yaml import YAML
# from ruamel.yaml.scalarstring import PreservedScalarString
import yaml
import json

class Agent:
    def __init__(self, config):
        self.char = config["char"]
        self.user = config["user"]
        self.char_settings = config["char_settings"]
        self.char_personalities = config["char_personalities"]
        self.message_example = config["message_example"]
        self.mask = config["mask"]

        self.is_data_base = config["is_data_base"]
        self.data_base_thresholds = config["data_base_thresholds"]
        self.data_base_depth = config["data_base_depth"]

        self.is_long_mem = config["is_long_mem"]
        self.is_check_memorys = config["is_check_memorys"]
        self.mem_thresholds = config["mem_thresholds"]

        self.is_core_mem = config["is_core_mem"]
        self.llm_config = config["llm"]
        

        # 创建上下文
        # self.msg_data = list[dict[str, str]]
        self.msg_data = []
        # 上下文缓存
        self.msg_data_tmp = []
        try:
            with open(f"./data/agents/{self.char}/history.yaml", "r", encoding="utf-8") as f:
                self.msg_data = yaml.safe_load(f)["messages"]
        except:
            pass

        # 载入提示词
        self.prompt = []
        # self.prompt.append({"role": "system", "content": f"当前系统时间：{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}"})
        # tt = '''6. 注意输出文字的时候，将口语内容使用""符号包裹起来，并且优先输出口语内容，其他文字使用()符号包裹。'''
        self.long_mem_prompt = prompt.long_mem_prompt
        self.data_base_prompt = prompt.data_base_prompt
        self.core_mem_prompt = prompt.core_mem_prompt
        if self.char_settings:
            self.system_prompt = prompt.system_prompt.replace("{{char}}", self.char).replace("{{user}}", self.user)
            self.char_setting_prompt = prompt.char_setting_prompt.replace("{{char_setting_prompt}}", self.char_settings).replace("{{char}}", self.char).replace("{{user}}", self.user)
            self.prompt.append({"role": "system", "content": self.system_prompt})
            self.prompt.append({"role": "system", "content": self.char_setting_prompt})
        if self.char_personalities:
            self.char_Personalities_prompt = prompt.char_Personalities_prompt.replace("{{char_Personalities_prompt}}", self.char_personalities).replace("{{char}}", self.char).replace("{{user}}", self.user)
            self.prompt.append({"role": "system", "content": self.char_Personalities_prompt})
            # self.prompt += self.char_Personalities_prompt + "\n\n"
        if self.mask:
            self.mask_prompt = prompt.mask_prompt.replace("{{mask_prompt}}", self.mask).replace("{{char}}", self.char).replace("{{user}}", self.user)
            self.prompt.append({"role": "system", "content": self.mask_prompt})
            # self.prompt += self.mask_prompt + "\n\n"
        if self.message_example:
            self.message_example_prompt = prompt.message_example_prompt.replace("{{message_example}}", self.message_example).replace("{{user}}", self.user).replace("{{char}}", self.char)
            self.prompt.append({"role": "system", "content": self.message_example_prompt})
            # self.prompt += self.message_example_prompt + "\n\n"
        if config["prompt"]:
            self.prompt.append({"role":  "system", "content": config["prompt"]})
            # self.prompt += config["prompt"]

        # 创建系统时间戳
        self.tt = int(time.time())

        # 创建数据存储文件夹
        os.path.exists(f"./data/agents/{self.char}/memorys") or os.makedirs(f"./data/agents/{self.char}/memorys")
        os.path.exists(f"./data/agents/{self.char}/data_base") or os.makedirs(f"./data/agents/{self.char}/data_base")

        # 加载角色记忆
        if self.is_long_mem:
            config = {
                "char": self.char,
                "user": self.user,
                "is_check_memorys": self.is_check_memorys,
                "thresholds": self.mem_thresholds
            }
            self.Memorys = long_mem.Memorys(config)
        
        # 加载核心记忆
        if self.is_core_mem:
            self.Core_mem = core_mem.Core_Mem({"char": self.char, "user": self.user})

        # 载入知识库
        if self.is_data_base:
            self.DataBase = data_base.DataBase(
                char=self.char,
                thresholds=self.data_base_thresholds,
                top_k=self.data_base_depth
            )

    # 知识库内容检索
    def get_data(self, msg: str, res_msg: list) -> str:
        msg_list = jionlp.split_sentence(msg, criterion='fine')
        res_ = self.DataBase.search(msg_list)
        if res_ != "":
            res_msg.append(res_)

    # 提取、插入核心记忆
    def insert_core_mem(self, msg2: str, msg3: str, msg1: str):
        mmsg = prompt.get_core_mem.replace("{{memories}}", json.dumps(self.Core_mem.mems[-100:], ensure_ascii=False))
        if self.msg_data[-1]["role"] != "assistant":
            return
        re_msg = "对话内容：助手：" + msg1 + "\n用户：" + msg2 + "\n助手：" + msg3
        key = self.llm_config["key"]
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.llm_config["model"],
            "messages": [
                {"role": "system", "content": mmsg},
                {"role": "user", "content": re_msg}
            ]
        }
        try:
            res = requests.post(self.llm_config["api"], json=data, headers=headers, timeout=15)
            res_msg = res.json()["choices"][0]["message"]["content"]
            mem_list = ast.literal_eval(jionlp.extract_parentheses(res_msg, "[]")[0].replace(" ", "").replace("\n", ""))
            if len(mem_list) > 0:
                self.Core_mem.add_memory(mem_list)
        except:
            return
    # 提取记忆摘要，记录长期记忆
    # def add_memory1(self, data: list, t_n: int):
    #     mmsg = prompt.get_mem_tag_prompt
    #     res_msg = "用户：" + data[-2]["content"] + "\n助手：" + data[-1]["content"]
    #     res_body = {
    #         "model": self.llm_config["model"],
    #         "messages": [
    #             {"role": "system", "content": mmsg},
    #             {"role": "user", "content": res_msg}
    #         ]
    #     }
    #     key = self.llm_config["key"]
    #     headers = {
    #         "Authorization": f"Bearer {key}",
    #         "Content-Type": "application/json"
    #     }
    #     res_tag = ""
    #     try:
    #         res = requests.post(self.llm_config["api"], json=res_body, headers=headers, timeout=15)
    #         res = res.json()["choices"][0]["message"]["content"]
    #         res = jionlp.remove_html_tag(res).replace(" ", "").replace("\n", "").replace("摘要:", "").replace("摘要：", "")
    #         if res.find("日常闲聊") != -1:
    #             res_tag = res
    #         else:
    #             res_tag = "日常闲聊"
    #     except:
    #         res_tag = "日常闲聊"
    #     t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_n))
    #     m1 = data[-2]["content"]
    #     m2 = data[-1]["content"]
    #     m_data = {
    #         "t_n": t_n,
    #         "text_tag": res_tag,
    #         "msg": f"时间：{t_str}\n{self.user}：{m1}\n{self.char}：{m2}"
    #     }
    #     self.Memorys.add_memory(m_data)     

    # 获取发送到大模型的上下文
    def get_msg_data(self, msg: str) -> list:
        # index = len(self.msg_data) - 1
        # g_t = Thread(target=self.insert_core_mem, args=(msg, index,))
        # g_t.daemon = True
        # g_t.start()
        
        ttt = int(time.time())
        self.tt = ttt
        t_n = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ttt))
        self.prompt[0] = {"role": "system", "content": f"当前现实世界时间：{t_n}"}
        # self.tmp_mem = f"时间：{t_n}\n{self.user}：{msg.strip()}\n"
        t_list = []
        data_base = []
        mem_msg = []
        res_msg = []
        core_mem = []
        res_msg += self.prompt

        # 检索世界书
        if self.is_data_base:
            tt = Thread(target=self.get_data, args=(msg, data_base, ))
            tt.daemon = True
            t_list.append(tt)
            tt.start()

        # 搜索记忆
        if self.is_long_mem:
            tt = Thread(target=self.Memorys.get_memorys, args=(msg, mem_msg, t_n))
            tt.daemon = True
            t_list.append(tt)
            tt.start()

        # 搜索核心记忆
        if self.is_core_mem:
            tt = Thread(target=self.Core_mem.find_mem, args=(msg, core_mem, ))
            tt.daemon = True
            t_list.append(tt)
            tt.start()

        # 等待查询结果
        for tt in t_list:
            tt.join()
        
        # 合并上下文、世界书、记忆信息
        # self.msg_data.append(
        #     {
        #         "role": "user",
        #         "content": msg
        #     }
        # )
        self.msg_data_tmp = []
        if self.is_data_base and data_base:
            # self.msg_data.append({"role": "system", "content": self.data_base_prompt.replace("{{data_base}}", data_base[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
            self.msg_data_tmp.append({"role": "system", "content": self.data_base_prompt.replace("{{data_base}}", data_base[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
        if self.is_core_mem and core_mem:
            # self.msg_data.append({"role": "system", "content": self.core_mem_prompt.replace("{{core_mem}}", core_mem[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
            self.msg_data_tmp.append({"role": "system", "content": self.core_mem_prompt.replace("{{core_mem}}", core_mem[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
        if self.is_long_mem and mem_msg:
            # self.msg_data.append({"role": "system", "content": self.long_mem_prompt.replace("{{memories}}", mem_msg[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
            self.msg_data_tmp.append({"role": "system", "content": self.long_mem_prompt.replace("{{memories}}", mem_msg[0]).replace("{{user}}", self.user).replace("{{char}}", self.char)})
        # self.msg_data.append({"role": "system", "content": f"当前现实世界时间：{t_n}"})
        # 合并上下文、世界书、记忆信息
        self.msg_data_tmp.append(
            {
                "role": "user",
                "content": msg
            }
        )
        # self.msg_data_tmp = tmp_msg_data
        return res_msg + self.msg_data[-30:] + self.msg_data_tmp

    # 刷新上下文内容
    def add_msg(self, msg: str):
        # def find_last(key: str):
        #     for i in range(len(self.msg_data)):
        #         if self.msg_data[len(self.msg_data)-1-i]["role"] == key:
        #             return self.msg_data[len(self.msg_data)-1-i]["content"]

        self.msg_data_tmp.append(
            {
                "role": "assistant",
                "content": msg
            }
        )
        msg_data_tmp = self.msg_data_tmp.copy()
        # t_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.tt))
        # m1 = find_last("user")
        # m2 = self.msg_data[-1]["content"]
        m1 = msg_data_tmp[-2]["content"]
        # m2 = msg_data_tmp[-1]["content"]

        try:
            ttt1 = Thread(target=self.insert_core_mem, args=(m1, self.msg_data_tmp[-1]["content"], self.msg_data[-1]["content"]))
            ttt1.daemon = True
            ttt1.start()
        except:
            print(f"[错误]{self.msg_data_tmp}")

        ttt2 = Thread(target=self.Memorys.add_memory1, args=(msg_data_tmp, self.tt, self.llm_config))
        ttt2.daemon = True
        ttt2.start()

        # msg1 = f"时间：{t_str}\n"
        # msg1 += f"{self.user}：{m1}\n"
        # msg1 += f"{self.char}：{m2}"
        # M_data = {
        #     "t_n": self.tt,
        #     "msg": msg1
        # }
        # self.add_memory(msg_data_tmp, self.tt, self.llm_config)
        self.msg_data += self.msg_data_tmp
        self.msg_data = self.msg_data[-60:]
        write_data = {
            "messages": self.msg_data[-60:]
        }
        with open(f"./data/agents/{self.char}/history.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(write_data, f, allow_unicode=True)
