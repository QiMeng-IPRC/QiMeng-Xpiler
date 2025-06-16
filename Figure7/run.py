import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Read CSV data, handle multi-level column headers.
df = pd.read_csv('2.csv', header=[0, 1])

# 2. Print the original multi-level column names for debugging purposes.
print("原始多级列名:")
print(df.columns)

# 3. Flatten multi-level column names into single-level column names, ignoring the 'Unnamed' parts.
new_columns = []
last_main = ''

for col in df.columns:
    if 'Unnamed' not in col[0]:
        last_main = col[0]
        if 'Unnamed' in str(col[1]) or pd.isna(col[1]):
            new_columns.append(col[0])
        else:
            new_columns.append(f"{col[0]} {col[1]}")
    else:
        if 'Unnamed' in str(col[1]) or pd.isna(col[1]):
            new_columns.append(col[0])
        else:
            new_columns.append(f"{last_main} {col[1]}")

df.columns = new_columns

# 4. Print the flattened column names for confirmation.
print("\n平坦化后的列名:")
print(df.columns)

# 5. Set global font size and chart parameters
plt.rcParams.update({
    'font.size': 16,             # Overall font size
    'axes.labelsize': 18,        # Axis label font size
    'xtick.labelsize': 14,       # X-axis scale font size
    'ytick.labelsize': 14,       # Y-axis scale font size
    'legend.fontsize': 20,       # Legend font size (increase)
    'figure.figsize': (20, 15)   # Chart size
})

# 6. Create four subplots, arranged vertically, sharing the X-axis, with constrained_layout set to False.
fig, axes = plt.subplots(4, 1, figsize=(20, 15), sharex=True, constrained_layout=False)

fig.patch.set_facecolor('white')  # Set the background to white.

# 7. Create a Boolean mask that includes all rows containing "Overall".
mask = df['Type'].notna()

# 8. Obtain the operator type, including "Overall".
operators = df.loc[mask, 'Type']
x = np.arange(len(operators))  # The number of operators

# 9. Add a gap before the last set of data.
x_new = x.copy()
if len(x_new) > 1:
    x_new[-1] += 1  # Move the last set of data one unit to the right to create a gap.

# 10. The width of the bar chart
width = 0.35

# 11. PyTorch performance is set to 1.
pytorch_perf = [1] * len(operators)

# 12. Define the conversion types, corresponding column names, and conversion names.
transitions = ['C->CUDA', 'CUDA->BANG', 'CUDA->HIP', 'CUDA->C']
transition_names = [
    'C with VNNI → CUDA C',
    'CUDA C → BANG C',
    'CUDA C → HIP',
    'CUDA C → C with VNNI'
]

# 13. Initialize legend handles and labels.
handles = []
labels = []

# 14. Iteratively draw each subplot.
for idx, (transition, transition_name) in enumerate(zip(transitions, transition_names)):
    ax = axes[idx]

    # 15. Construct column names
    corrected_cases_col = f'{transition} Corrected cases'
    speedup_col = f'{transition} SpeedUp Over Pytorch'

    # 16. Check if the column exists to avoid KeyError.
    if corrected_cases_col not in df.columns or speedup_col not in df.columns:
        print(f'列 "{corrected_cases_col}" 或 "{speedup_col}" 未找到。请检查列名。')
        continue

    # 17. Obtain the corresponding conversion for Corrected cases and SpeedUp Over Pytorch, including the "Overall" row.
    corrected_cases = df.loc[mask, corrected_cases_col].fillna(0).values
    speedup = df.loc[mask, speedup_col].fillna(0).values

    # 18. Check if the data lengths match.
    if len(speedup) != len(x_new):
        print(f'警告: "{transition} SpeedUp Over Pytorch" 的长度 {len(speedup)} 与 operators 的长度 {len(x_new)} 不匹配。')
        continue

    # 19. Draw a bar chart.
    bars1 = ax.bar(x_new - width / 2, pytorch_perf, width, label='PyTorch', color='orange', zorder=2)  # PyTorch is orange.
    bars2 = ax.bar(x_new + width / 2, speedup, width, label='QiMeng-Xpiler', color='purple', zorder=2)        # Falcon is for purple.

    # 20. Set the X-axis scale.
    ax.set_xticks(x_new)
    ax.set_xticklabels(operators, rotation=45, ha='right')

    # 21. Add grid lines
    ax.yaxis.grid(True, linestyle='--', linewidth=0.5)

    # 22. Create a second Y-axis for plotting the line chart.
    ax2 = ax.twinx()
    line = ax2.plot(x_new, corrected_cases, color='red', marker='o', label='Corrected Cases', zorder=3)[0]  # Corrected Cases are in red.

    # 23. Set a fixed range for the right y-axis and configure the scale.
    ax2.set_ylim(0, 10)
    ax2.set_yticks([0, 8])

    # 24. Remove data labels from the line chart points.
    # (No comment)

    # 25. Add subfigure labels and transform names.
    ax.text(0.5, 1.05, f"({chr(97+idx)}) {transition_name}", transform=ax.transAxes, fontsize=20, fontweight='bold', ha='center')

    # 26. Collect legend handles only in the first subplot.
    if idx == 0:
        handles.extend([bars1[0], bars2[0], line])
        labels.extend(['PyTorch', 'QiMeng-Xpiler', 'Corrected Cases'])

# 27. Create a unified legend, positioned at the top center of the entire chart, and increase the font size.
fig.legend(handles=handles, labels=labels, loc='upper center', fontsize=24, ncol=3)

# 28. Add shared y-axis titles on the left and right sides, ensuring they are positioned outside the chart.
fig.text(0.02, 0.5, 'Normalized Performance', va='center', rotation='vertical', fontsize=18, ha='center')
fig.text(0.98, 0.5, 'Corrected Cases', va='center', rotation='vertical', fontsize=18, ha='center')

# 29. Adjust the overall layout to ensure that legends and subfigure labels are not obscured.
# Maintain the user-specified margins.
fig.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.2, hspace=0.3)

# 30. Display the chart.
#plt.show()
fig.savefig("Figure_7-3_new.pdf")
