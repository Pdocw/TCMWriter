import dspy
from typing import Union
from modules.utils import clean_up_records, load_str, clean_up_polish_records, clean_up_four_records
from modules.tcm_expert import TCMExpert
import os
from time import sleep

class PolishRecordsNotes(dspy.Signature):
    """您是一个经验丰富的老中医，请根据所给的中医医案、四部分中医按语、收集到的相关信息和草稿按语，润色草稿按语撰写一份内容更加丰富合理的中医按语，尽可能完整覆盖医案和四部分按语中的信息。 请遵照以下格式用中文输出：
        1. 以 "润色后的按语："开头的格式，只包含一段内容。不要分行。
        2. 不要包含其他任何信息。
    """
    medicine_medical_records = dspy.InputField(prefix="您需要撰写按语的中医医案：", format=str)
    four_part_records_notes = dspy.InputField(prefix="四个部分按语：", format=str)
    draft_records_notes = dspy.InputField(prefix="您需要润色的草稿按语：", format=str)
    info = dspy.InputField(prefix='收集到的相关信息：\n', format=str)
    records_notes = dspy.OutputField(prefix="润色后的按语：\n", format=str)

class PolishRecordsNotesWithoutInfo(dspy.Signature):
    """您是一个经验丰富的老中医，请根据所给的中医医案、四部分中医按语和草稿按语，润色草稿按语撰写一份内容更加丰富合理的中医按语。 请遵照以下格式用中文输出：
        1. 以 "润色后的按语："开头的格式，只包含一段内容。不要分行。
        2. 不要包含其他任何信息。
    """
    medicine_medical_records = dspy.InputField(prefix="您需要撰写按语的中医医案：", format=str)
    four_part_records_notes = dspy.InputField(prefix="四个部分按语：", format=str)
    draft_records_notes = dspy.InputField(prefix="您需要润色的草稿按语：", format=str)
    records_notes = dspy.OutputField(prefix="润色后的按语：\n", format=str)
    
    
class PolishRecordsNotesWithoutFourpart(dspy.Signature):
    """您是一个经验丰富的老中医，请根据所给的中医医案、收集到的相关信息和草稿按语，润色草稿按语撰写一份内容更加丰富合理的中医按语。 请遵照以下格式用中文输出：
        1. 以 "润色后的按语："开头的格式，只包含一段内容。不要分行。
        2. 不要包含其他任何信息。
    """
    medicine_medical_records = dspy.InputField(prefix="您需要撰写按语的中医医案：", format=str)
    draft_records_notes = dspy.InputField(prefix="您需要润色的草稿按语：", format=str)
    info = dspy.InputField(prefix='收集到的相关信息：\n', format=str)
    records_notes = dspy.OutputField(prefix="润色后的按语：\n", format=str)
    
class FourPartRecordsNotes(dspy.Signature):
    """您是一个经验丰富的老中医，请根据所给的中医医案和收集到的相关信息，生成以下四个部分的按语： ①病情概况和病因分析，②治疗方法和原则总结，③药物处方或疗法的说明，④疗效评价、注意事项和医案心得。 请遵照以下格式用中文输出：
        1. 第一行以" ①病情概况和病因分析："开头，第二行以"②治疗方法和原则总结："开头，第三行以"③药物处方或疗法的说明："开头，第四行以"④疗效评价、注意事项和医案心得："开头。
        2. 只给出四行内容，不要包含其他任何信息。
    """
    medicine_medical_records = dspy.InputField(prefix="您需要撰写按语的中医医案：", format=str)
    info = dspy.InputField(prefix='收集到的相关信息：\n', format=str)
    four_part_records_notes = dspy.OutputField(prefix="四个部分按语：\n", format=str)
    
class DirectWriteRecordsNotes(dspy.Signature):
    """请为所给医案撰写一份按语，请遵照以下格式用中文输出：
        1. 以 "按语："开头的格式，只包含一段内容。
        2. 不要包含其他任何信息。
    """
    medicine_medical_records = dspy.InputField(prefix="您需要撰写按语的中医医案：", format=str)
    records_notes = dspy.OutputField(prefix="按语：\n", format=str)


class WriteRecords(dspy.Module):
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel]):
        super().__init__()
        self.draft_records_notes = dspy.Predict(DirectWriteRecordsNotes)
        self.four_part_records_notes = dspy.Predict(FourPartRecordsNotes)
        self.polish_records_notes = dspy.Predict(PolishRecordsNotes)
        self.polish_records_notes_without_info = dspy.Predict(PolishRecordsNotesWithoutInfo)
        self.polish_records_notes_without_four = dspy.Predict(PolishRecordsNotesWithoutFourpart)
        self.tcm_expert = TCMExpert(engine=engine)
        self.engine = engine
    
    
    def forward(self, medicine_medical_records, records_idx):
        with dspy.settings.context(lm=self.engine):
            draft_records_notes = clean_up_records(self.draft_records_notes(medicine_medical_records=medicine_medical_records).records_notes)
            info_path = os.path.join('../results/info', 'info_' + str(records_idx) + '.txt')
            if os.path.isfile(info_path):
                info = load_str(info_path)
            else:
                expert_output = self.tcm_expert(medicine_medical_records=medicine_medical_records)
                info = expert_output.info
            four_part_records_notes = clean_up_four_records(self.four_part_records_notes(medicine_medical_records=medicine_medical_records,info=info).four_part_records_notes)
            without_info_records_notes = clean_up_polish_records(self.polish_records_notes_without_info(medicine_medical_records=medicine_medical_records,four_part_records_notes=four_part_records_notes,draft_records_notes=draft_records_notes).records_notes)
            without_four_records_notes = clean_up_polish_records(self.polish_records_notes_without_four(medicine_medical_records=medicine_medical_records,info=info,draft_records_notes=draft_records_notes).records_notes)
            records_notes = clean_up_polish_records(self.polish_records_notes(medicine_medical_records=medicine_medical_records,four_part_records_notes=four_part_records_notes,info=info,draft_records_notes=draft_records_notes).records_notes)
            
        return dspy.Prediction(draft_records_notes=draft_records_notes,four_part_records_notes=four_part_records_notes,records_notes=records_notes,info=info,without_info_records_notes=without_info_records_notes,without_four_records_notes=without_four_records_notes)