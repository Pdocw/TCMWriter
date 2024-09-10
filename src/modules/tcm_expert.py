import logging
import dspy
import os
import re
from urllib.parse import urlparse
from typing import Callable, Union, List
from modules.utils import get_meta_content
import requests

class MedicalToEntity(dspy.Signature):
    """您是一个经验丰富的老中医，请您根据所给的中医医案判断以下该医案属于的疾病种类。请输出百度百科关于该疾病的url, 请遵照以下格式：
        1. 以 "https://baike.baidu.com/item/"  + "疾病种类名称" 的格式。
        2. 不要包含其他任何信息。
    """

    medicine_medical_records = dspy.InputField(prefix="中医医案：", format=str)
    url = dspy.OutputField(prefix="百度百科关于该疾病的url：\n", format=str)

class GenDescription(dspy.Signature):
    """您是一个经验丰富的老中医，请您为所给的疾病生成一段详细的描述。
    """
    disease = dspy.InputField(prefix="疾病名称：", format=str)
    description = dspy.OutputField(prefix="您是一个经验丰富的老中医，请您为所给的疾病生成一段详细的描述：\n", format=str)
class TCMExpert(dspy.Module):

    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.generate_url = dspy.Predict(MedicalToEntity)
        self.gen_description = dspy.Predict(GenDescription)
        self.engine = engine

    def forward(self, medicine_medical_records: str):
        with dspy.settings.context(lm=self.engine):
            url = self.generate_url(medicine_medical_records=medicine_medical_records).url
            if url[-2:] == "心悸":
                url += "病"
            info = get_meta_content(url)
            if info == None:
                disease = url.replace("https://baike.baidu.com/item/", "")
                info = self.gen_description(disease=disease).description

        return dspy.Prediction(info=info)