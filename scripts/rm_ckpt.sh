#!/bin/bash

# 指定路径和需要保留的编号
ckpt_dir="/data/fsq/GUI-R1_exp/GUI-R1_ckpt"
save_id="80"

# 遍历 ckpt_dir 下的 global_step_* 文件夹
for dir in "$ckpt_dir"/global_step_*; do
  [ -e "$dir" ] || continue

  step_id="${dir##*_}"

  if [ "$step_id" != "$save_id" ]; then
    echo "删除: $dir"
    rm -rf "$dir"
  else
    echo "保留: $dir"
  fi
done
