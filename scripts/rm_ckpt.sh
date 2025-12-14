#!/bin/bash

# 指定路径和需要保留的编号列表（用空格分隔）
ckpt_dir="/mnt/Shared_05_disk/fsq/gui_agent_exp/gui-r1/mind2web_gt_grpo_qwen2_5_vl_3b_105"
save_ids=("200" "570")  # 这里可以添加多个编号

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
