python eval_records_without.py --input-path ../TCMMMR/records.xlsx --type info --gen-records-path ../results/Qwen2/polish_records_notes_without_info/
# 检查上一个命令是否成功
if [ $? -ne 0 ]; then
  echo "glm4 模型执行失败，终止脚本。"
  exit 1
fi

python eval_records_without.py --input-path ../TCMMMR/records.xlsx --type four --gen-records-path ../results/Qwen/polish_records_notes_without_four/
# 检查上一个命令是否成功
if [ $? -ne 0 ]; then
  echo "glm4 模型执行失败，终止脚本。"
  exit 1
fi

