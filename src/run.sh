# python -m scripts.run --input-source file --input-path ../TCMMMR/records.xlsx
# # 检查上一个命令是否成功
# if [ $? -ne 0 ]; then
#   echo "gpt 模型执行失败，终止脚本。"
#   exit 1
# fi




# python -m scripts.run --model-name glm --input-source file --input-path ../TCMMMR/records.xlsx
# # 检查上一个命令是否成功
# if [ $? -ne 0 ]; then
#   echo "glm4 模型执行失败，终止脚本。"
#   exit 1
# fi


python -m scripts.run --model-name Qwen --input-source file --input-path ../TCMMMR/records.xlsx
# 检查上一个命令是否成功
if [ $? -ne 0 ]; then
  echo "Qwen2 模型执行失败，终止脚本。"
  exit 1
fi