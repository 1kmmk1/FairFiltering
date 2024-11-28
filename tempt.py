# import math
# # def trade_off(acc:list, wga:list):
# #     baseline_acc = [98.1, 95.5, 91.9, 82.1]
# #     baseline_wga = [75.3, 45.9, 66.7, 68.4]
# #     for i in range(len(acc)):
# #         td = (wga[i] - baseline_wga[i]) / (baseline_acc[i] - acc[i])
# #         print(td, end="||")
    
    
# # jtt_acc = [93.3, 88.0, 78.6, 91.1];jtt_wga = [86.7, 81.1, 78.6, 69.3]
# # cnc_acc = [90.9, 89.9, 0, 81.7]; cnc_wga = [88.5, 88.5, 0, 68.9]
# # ssa_acc = [92.2, 92.8, 79.9, 76.6]; ssa_wga = [89.0, 89.8, 76.6, 69.9]
# # dfr_acc = [94.2, 91.3, 82.1, 87.2];dfr_wga = [92.9, 82.1, 74.7, 70.1]
# # self_acc = [94.0, 91.7, 79.1, 81.2];self_wga = [93.0, 83.9, 79.1, 70.7]

# # our_acc = [97.1, 91.2, 90.1, 0];our_wga = [93.3, 81.0, 72.02, 0]

# # trade_off(our_acc, our_wga)\
    
    
# import matplotlib.pyplot as plt
# import pandas as pd

# datasets = ['Waterbirds', 'CelebA', 'CivilComments', 'MultiNLI']
# baseline = [74.9, 50.0, 66.9, 68.6]
# Filtering = [84.9, 78.9, 81.9, 71.2]
# no_learn = [81.8, 78.9, 72.2, 61.5]

# import matplotlib.pyplot as plt
# import numpy as np

# # 데이터 예시
# datasets = ['Waterbirds', 'CelebA', 'CivilComments', 'MultiNLI']

# # 설정
# num_datasets = len(datasets)
# num_methods = 3
# width = 0.25  # 막대 너비
# group_spacing = 1.5  # 그룹 간 간격

# # x축 위치 계산
# x = np.arange(num_datasets) * group_spacing
# x_baseline = x - width
# x_filtering = x
# x_no_learn = x + width

# fig, ax = plt.subplots(figsize=(12, 7))

# # 막대 그리기
# rects_baseline = ax.bar(x_baseline, baseline, width, label='Baseline', color='gray')
# rects_filtering = ax.bar(x_filtering, Filtering, width, label='Learnable Filter', color='skyblue')
# rects_no_learn = ax.bar(x_no_learn, no_learn, width, label='Random Filter', color='orange')

# # 레이블 및 제목 설정
# ax.set_ylabel('Worst Group Accuracy', fontsize=18)
# # ax.set_title('Comparison of Methods Across Datasets')  # 필요 시 주석 해제
# ax.set_xticks(x)
# ax.set_xticklabels(datasets, fontsize=18)
# ax.legend(fontsize=20)

# ax.set_ylim(40, 100)  # 필요에 따라 범위를 조정하세요

# # 값 라벨 추가
# def autolabel(rects, fontsize=14):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.1f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 포인트 위
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=fontsize)

# autolabel(rects_baseline)
# autolabel(rects_filtering)
# autolabel(rects_no_learn)

# # 간격 추가를 위한 레이아웃 조정
# plt.tight_layout()
# plt.savefig("test2.png")
# plt.show()



import os
import numpy as np
import matplotlib.pyplot as plt

# Define the dataset folders
datasets = [
    'baseline/Waterbirds_baseline_seed1001',
    'baseline/CelebA_baseline_seed1001',
    'baseline/CivilComments_baseline_seed1001',
    'baseline/MultiNLI_baseline_seed1001'
]

# Define the dropout ratios (assuming 0.2 to 1.0 with a step of 0.1 for 9 files)
dropout_ratios = [round(0.2 + 0.1 * i, 1) for i in range(8)]  # [0.2, 0.3, ..., 1.0]

# Initialize a dictionary to store min values for each dataset
min_values_dict = {dataset: [] for dataset in datasets}

# Iterate through each dataset and dropout ratio to extract min values
for dataset in datasets:
    for dropout in dropout_ratios:
        # Construct the file path
        file_name = f'dropout_test_result_{dropout}.npy'
        file_path = os.path.join('log', dataset, file_name)
        
        # Check if the file exists
        if os.path.isfile(file_path):
            try:
                # Load the .npy file
                data = np.load(file_path)
                
                # Compute the minimum value
                min_val = np.min(data)
                
                # Append the min value to the corresponding dataset's list
                min_values_dict[dataset].append(min_val)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                min_values_dict[dataset].append(np.nan)  # Use NaN for missing/errored data
        else:
            print(f"File not found: {file_path}")
            min_values_dict[dataset].append(np.nan)  # Use NaN for missing files

# Plotting
plt.figure(figsize=(12, 7))

# Define colors for each dataset
colors = ['skyblue', 'orange', 'green', 'red']
markers = ['o', 's', '^', 'D']  # Different markers for clarity

# Iterate through datasets to plot each
for idx, dataset in enumerate(datasets):
    min_values = min_values_dict[dataset]
    plt.plot(dropout_ratios, min_values, marker=markers[idx], color=colors[idx],
             label=os.path.basename(dataset))  # Use folder name as label

# Customize the plot
plt.xlabel('Dropout Ratio', fontsize=16)
plt.ylabel('Minimum Value (np.min())', fontsize=16)
plt.title('Minimum Values Across Dropout Ratios for Each Dataset', fontsize=18)
plt.xticks(dropout_ratios, fontsize=14)
plt.yticks(fontsize=14)
plt.legend(title='Datasets', fontsize=14, title_fontsize=16)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()

# Save and show the plot
plt.savefig("dropout_min_values_comparison.png")
plt.show()
