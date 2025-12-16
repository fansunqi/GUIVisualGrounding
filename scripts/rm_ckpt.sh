#!/bin/bash

# 指定路径和需要保留的编号列表（用空格分隔）
ckpt_dir="/data/fsq/GUI-R1_exp/mind2web_ws_grpo_qwen2_5_vl_3b_r1wsv2"
save_ids=("25" "50" "75" "100" "125" "150" "175" "200" "225")  # 这里可以添加多个编号

# 遍历 ckpt_dir 下的 global_step_* 文件夹
for dir in "$ckpt_dir"/global_step_*; do
  [ -e "$dir" ] || continue

  step_id="${dir##*_}"

  keep=0
  for id in "${save_ids[@]}"; do
    if [ "$step_id" = "$id" ]; then
      keep=1
      break
    fi
  done

  if [ $keep -eq 1 ]; then
    echo "保留: $dir"
  else
    echo "删除: $dir"
    rm -rf "$dir"
  fi
done
