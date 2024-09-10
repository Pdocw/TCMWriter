import functools
import operator
import threading
import dspy
import re
import os
import toml
import sys
import requests
from bs4 import BeautifulSoup
from typing import Optional, Union, Literal, Any, List

class MyOpenAIModel(dspy.OpenAI):
    """A wrapper class for dspy.OpenAI to track token usage."""

    def __init__(
            self,
            model: str = "gpt-4",
            api_key: Optional[str] = None,
            api_provider: Literal["openai", "azure"] = "openai",
            api_base: Optional[str] = None,
            model_type: Literal["chat", "text"] = None,
            **kwargs
    ):
        super().__init__(model=model, api_key=api_key, api_provider=api_provider, api_base=api_base,
                         model_type=model_type, **kwargs)
        self._token_usage_lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def log_usage(self, response):
        """Log the total tokens from the OpenAI API response."""
        usage_data = response.get('usage')
        if usage_data:
            with self._token_usage_lock:
                self.prompt_tokens += usage_data.get('prompt_tokens', 0)
                self.completion_tokens += usage_data.get('completion_tokens', 0)

    def get_usage_and_reset(self):
        """Get the total tokens used and reset the token usage."""
        usage = {
            self.kwargs.get('model') or self.kwargs.get('engine'):
                {'prompt_tokens': self.prompt_tokens, 'completion_tokens': self.completion_tokens}
        }
        self.prompt_tokens = 0
        self.completion_tokens = 0

        return usage

    def __call__(
            self,
            prompt: str,
            only_completed: bool = True,
            return_sorted: bool = False,
            **kwargs,
    ) -> list[dict[str, Any]]:
        """Copied from dspy/dsp/modules/gpt3.py with the addition of tracking token usage."""

        assert only_completed, "for now"
        assert return_sorted is False, "for now"

        # if kwargs.get("n", 1) > 1:
        #     if self.model_type == "chat":
        #         kwargs = {**kwargs}
        #     else:
        #         kwargs = {**kwargs, "logprobs": 5}

        response = self.request(prompt, **kwargs)

        # Log the token usage from the OpenAI API response.
        self.log_usage(response)

        choices = response["choices"]

        completed_choices = [c for c in choices if c["finish_reason"] != "length"]

        if only_completed and len(completed_choices):
            choices = completed_choices

        completions = [self._get_choice_text(c) for c in choices]
        if return_sorted and kwargs.get("n", 1) > 1:
            scored_completions = []

            for c in choices:
                tokens, logprobs = (
                    c["logprobs"]["tokens"],
                    c["logprobs"]["token_logprobs"],
                )

                if "<|endoftext|>" in tokens:
                    index = tokens.index("<|endoftext|>") + 1
                    tokens, logprobs = tokens[:index], logprobs[:index]

                avglog = sum(logprobs) / len(logprobs)
                scored_completions.append((avglog, self._get_choice_text(c)))

            scored_completions = sorted(scored_completions, reverse=True)
            completions = [c for _, c in scored_completions]

        return completions

class GPTConfigs:
    def __init__(self):
        self.generate_records_lm = None

    def init_openai_model(
            self,
            openai_api_key: str,
            openai_type: Literal["openai", "azure"],
            api_base: Optional[str] = 'http://10.15.82.10:4000/v1/',
            temperature: Optional[float] = 1.0,
            top_p: Optional[float] = 0.9
    ):
        openai_kwargs = {
            'api_key': openai_api_key,
            'api_provider': openai_type,
            'temperature': temperature,
            'top_p': top_p,
            'api_base': api_base,
        }
        self.generate_records_lm = MyOpenAIModel(model='gpt-4',max_tokens=1024, **openai_kwargs)

    def set_conv_simulator_lm(self, model: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        self.generate_records_lm = model





class MyModel(dspy.HFModel):
    def __init__(self, model_path, temperature=1.0, top_p=0.9, max_tokens=500, no_repeat_ngram_size=5):
        super().__init__(model_path)
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def _generate(self, prompt, **kwargs):
        self.drop_prompt_from_output = True
        kwargs['temperature'] = self.temperature
        kwargs['top_p'] = self.top_p
        kwargs['max_tokens'] = self.max_tokens
        kwargs['no_repeat_ngram_size'] = self.no_repeat_ngram_size

        return super()._generate(prompt, **kwargs)

    
class LLMConfigs:
    def __init__(self):
        self.generate_records_lm = None
        
    def init_llama_model(
            self,
            model,
    ):
        self.generate_records_lm = model
        
def write_str(s, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(s)

def clean_up_polish_records(text):
    ref_index = text.find("润色后的按语：")
    if ref_index != -1:
        text = text[ref_index + len("润色后的按语："):]
    # 将文本按行分割
    lines = text.splitlines()
    
    # 过滤掉空行
    non_empty_lines = [line for line in lines if line.strip()]
    
    # 初始化一个列表，用于存储处理后的行
    processed_lines = []
    
    for line in non_empty_lines:
        if len(line) < 50:
            continue
        
        # 找到最后一个 "。" 的位置
        last_period_index = line.rfind("。")
        
        # 如果找到 "。"，则保留到最后一个 "。" 为止；否则保留整行
        if last_period_index != -1:
            line = line[:last_period_index + 1]
        
        last_period_index = line.rfind("按语结束")
        
        if last_period_index != -1:
            line = line[:last_period_index]
            
        processed_lines.append(line)
    
    # 将处理后的行重新连接成一个字符串
    cleaned_text = "\n".join(processed_lines)
    
    return cleaned_text

def clean_up_four_records(text):
    lines = text.splitlines()  # 将文本按行分割
    
    # 检查是否有包含所有标记的行
    for line in lines:
        if all(marker in line for marker in ["①", "②", "③", "④"]):
            return line  # 如果找到包含所有标记的行，直接返回这行
    
    for line in lines:
        if all(marker in line for marker in ["病情概況", "治疗方法", "药物处方", "医案心得"]):
            return line  # 如果找到包含所有标记的行，直接返回这行
    index_1 = -1
    index_4 = -1
    
    for i, line in enumerate(lines):
        if line.startswith("①"):
            index_1 = i
        if line.startswith("④"):
            index_4 = i
    
    if index_1 != -1 and index_4 != -1:
        difference = index_4 - index_1
    else:
        difference = -1
    

    result_lines = []

    # 用于跟踪每个标记行是否已经添加
    added_markers = {
        "①": False,
        "②": False,
        "③": False,
        "④": False
    }

    if difference == 3:
        for line in lines:
            if line.startswith("①") and not added_markers["①"]:
                result_lines.append(line)
                added_markers["①"] = True
            elif line.startswith("②") and not added_markers["②"]:
                result_lines.append(line)
                added_markers["②"] = True
            elif line.startswith("③") and not added_markers["③"]:
                result_lines.append(line)
                added_markers["③"] = True
            elif line.startswith("④") and not added_markers["④"]:
                result_lines.append(line)
                added_markers["④"] = True

    elif difference == 6:
        for i, line in enumerate(lines):
            if line.startswith("①") and not added_markers["①"]:
                result_lines.append(line)
                added_markers["①"] = True
                if i + 1 < len(lines):
                    result_lines.append(lines[i + 1])
            elif line.startswith("②") and not added_markers["②"]:
                result_lines.append(line)
                added_markers["②"] = True
                if i + 1 < len(lines):
                    result_lines.append(lines[i + 1])
            elif line.startswith("③") and not added_markers["③"]:
                result_lines.append(line)
                added_markers["③"] = True
                if i + 1 < len(lines):
                    result_lines.append(lines[i + 1])
            elif line.startswith("④") and not added_markers["④"]:
                result_lines.append(line)
                added_markers["④"] = True
                if i + 1 < len(lines):
                    result_lines.append(lines[i + 1])



    if len(result_lines) == 0:
        if index_1 == -1 and index_4 == -1:
            for i, line in enumerate(lines):
                if "病情概況" in line:
                    index_1 = i
                if "医案心得" in line:
                    index_4 = i
        if index_1 != -1 and index_4 != -1:
            difference = index_4 - index_1
        else:
            difference = -1

        added_keywords = {
            "病情概況": False,
            "治疗方法": False,
            "药物处方": False,
            "医案心得": False
        }

        if difference == 3:
            for line in lines:
                if "病情概況" in line and not added_keywords["病情概況"]:
                    result_lines.append(line)
                    added_keywords["病情概況"] = True
                elif "治疗方法" in line and not added_keywords["治疗方法"]:
                    result_lines.append(line)
                    added_keywords["治疗方法"] = True
                elif "药物处方" in line and not added_keywords["药物处方"]:
                    result_lines.append(line)
                    added_keywords["药物处方"] = True
                elif "医案心得" in line and not added_keywords["医案心得"]:
                    result_lines.append(line)
                    added_keywords["医案心得"] = True

        elif difference == 6:
            for i, line in enumerate(lines):
                if "病情概況" in line and not added_keywords["病情概況"]:
                    result_lines.append(line)
                    added_keywords["病情概況"] = True
                    if i + 1 < len(lines):
                        result_lines.append(lines[i + 1])
                elif "治疗方法" in line and not added_keywords["治疗方法"]:
                    result_lines.append(line)
                    added_keywords["治疗方法"] = True
                    if i + 1 < len(lines):
                        result_lines.append(lines[i + 1])
                elif "药物处方" in line and not added_keywords["药物处方"]:
                    result_lines.append(line)
                    added_keywords["药物处方"] = True
                    if i + 1 < len(lines):
                        result_lines.append(lines[i + 1])
                elif "医案心得" in line and not added_keywords["医案心得"]:
                    result_lines.append(line)
                    added_keywords["医案心得"] = True
                    if i + 1 < len(lines):
                        result_lines.append(lines[i + 1])
    return "\n".join(result_lines)


def clean_up_records(text):
    ref_index = text.find("按语：")
    if ref_index != -1:
        text = text[ref_index + len("按语："):]
    # 将文本按行分割
    lines = text.splitlines()
    
    # 过滤掉空行
    non_empty_lines = [line for line in lines if line.strip()]
    
    # 初始化一个列表，用于存储处理后的行
    processed_lines = []

    for line in non_empty_lines:
        if len(line) < 50:
            continue
        # 找到最后一个 "。" 的位置
        last_period_index = line.rfind("。")
        
        # 如果找到 "。"，则保留到最后一个 "。" 为止；否则保留整行
        if last_period_index != -1:
            line = line[:last_period_index + 1]
        
        last_period_index = line.rfind("按语结束")
        
        if last_period_index != -1:
            line = line[:last_period_index]
            
        processed_lines.append(line)
    
    # 将处理后的行重新连接成一个字符串
    cleaned_text = "\n".join(processed_lines)
    
    return cleaned_text



def load_api_key(toml_file_path='../secrets.toml'):
    try:
        with open(toml_file_path, 'r') as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"File not found: {toml_file_path}", file=sys.stderr)
        return
    except toml.TomlDecodeError:
        print(f"Error decoding TOML file: {toml_file_path}", file=sys.stderr)
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)
        
def get_meta_content(url):
    
    response = requests.get(url)
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        
        meta_description = soup.find('meta', attrs={'name': 'description'})
        if meta_description:
            meta_content = meta_description.get('content')
        else:
            meta_content = None
        
        return meta_content
    else:
        return None

def load_str(path):
    with open(path, 'r', encoding='utf-8') as f:
        return '\n'.join(f.readlines())
