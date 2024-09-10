import os
from argparse import ArgumentParser

import pandas as pd
from tqdm import tqdm
from engine import DeepSearchRunnerArguments, DeepSearchRunner
from modules.utils import MyModel, LLMConfigs, load_api_key, GPTConfigs


def main(args):
    load_api_key()
    if args.model_name == "gpt":
        llm_configs = GPTConfigs()
        llm_configs.init_openai_model(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_type=os.getenv("OPENAI_API_TYPE"),
            api_base=os.getenv("API_BASE"),
        )
    elif args.model_name == "Qwen":
        llm_configs = LLMConfigs()
        model = MyModel("/home/lhw/code/models/Qwen2-7B-Instruct", max_tokens=1024)
        llm_configs.init_llama_model(model)
    elif args.model_name == "glm_4":
        llm_configs = LLMConfigs()
        model = MyModel("/home/lhw/code/models/glm-4-9b", max_tokens=1024)
        llm_configs.init_llama_model(model)

    engine_args = DeepSearchRunnerArguments(
        output_dir=args.output_dir,
        model_name=args.model_name,
    )

    runner = DeepSearchRunner(engine_args, llm_configs)

    if args.input_source == "console":
        medicine_medical_records = input("医案: ")
        runner.run(
            records_idx="console", medicine_medical_records=medicine_medical_records
        )

    else:
        data = pd.read_excel(args.input_path)
        for idx, row in tqdm(data.iterrows(), total=len(data)):
            medicine_medical_records = row["医案"]
            runner.run(
                records_idx=idx, medicine_medical_records=medicine_medical_records
            )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt")
    parser.add_argument(
        "--input-source",
        type=str,
        choices=["console", "file"],
        help="Where does the input come from.",
    )
    parser.add_argument("--input-path", type=str, help="Using csv file to store.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../results",
        help="Directory to store the outputs.",
    )
    main(parser.parse_args())
