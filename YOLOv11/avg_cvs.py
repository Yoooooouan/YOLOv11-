import pandas as pd
import glob
import os

# 1. 设置CSV文件所在目录路径（请修改为你的实际路径）
folder_path = '/Users/Lenovo/Desktop/detection_results'  # 替换为你的CSV文件夹路径
output_file = '/Users/Lenovo/Desktop/merged_average_results.csv'  # 汇总结果保存路径

# 2. 获取所有CSV文件路径
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

# 3. 读取并合并所有CSV文件
df_list = []
for file in all_files:
    try:
        df = pd.read_csv(file)
        df_list.append(df)
        print(f"已成功读取文件: {os.path.basename(file)}")
    except Exception as e:
        print(f"读取文件 {file} 时出错: {e}")

if not df_list:
    print("未找到任何CSV文件，请检查路径设置。")
else:
    # 使用 pd.concat 纵向合并所有DataFrame
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"\n所有文件合并完成！总数据量: {len(combined_df)} 行")

    # 4. 数据清洗（可选，但推荐）
    # 删除完全重复的行
    initial_count = len(combined_df)
    combined_df.drop_duplicates(inplace=True)
    print(f"移除 {initial_count - len(combined_df)} 行重复数据后，剩余 {len(combined_df)} 行。")

    # 5. 按'name'（目标类别）分组，并计算所有数值列的平均值
    # 关键点：`groupby` 后使用 `mean()`，默认会对所有可计算平均值的数值列进行运算[1,7](@ref)
    average_df = combined_df.groupby('name', as_index=False).mean(numeric_only=True)

    # 6. 保存结果
    average_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n汇总完成！所有数值列的平均值结果已保存至: {output_file}")

    # 打印结果预览
    print("\n结果预览（前10行）:")
    print(average_df.head(10))