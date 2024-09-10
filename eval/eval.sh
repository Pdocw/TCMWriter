python eval_records.py --input-path ../TCMMMR/records.xlsx --type draft --gen-records-path ../results/Qwen/draft_records_notes/
# 检查上一个命令是否成功
if [ $? -ne 0 ]; then
  exit 1
fi


python eval_records.py --input-path ../TCMMMR/records.xlsx --type polish --gen-records-path ../results/Qwen2/polish_records_notes/
# 检查上一个命令是否成功
if [ $? -ne 0 ]; then
  exit 1
fi
