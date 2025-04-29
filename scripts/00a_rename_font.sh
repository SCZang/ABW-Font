#!/bin/bash

# 设置初始序号
counter=0

# 遍历文件夹中的文件
for file in data/fonts/Font_Seen400/*; do
    # 判断是否为文件
    if test -f "$file"; then
        # 生成新的文件名
        new_name=$(printf "%03d_%s" $counter "$(basename "$file")")
        # 重命名文件
        mv "$file" "data/fonts/Font_Seen400/$new_name"
        # 更新序号
        counter=$((counter+1))
    fi
done
