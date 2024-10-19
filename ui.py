import matplotlib.pyplot as plt

# 指定输入文件的名称
input_file = "loss.txt"

# 初始化一个空列表来存储文件中的行
all_loss = []

# 打开文件以读取模式（'r'表示读取）
with open(input_file, 'r', encoding='utf-8') as file:
    # 读取文件中的所有行
    for line in file:
        # 去除每行末尾的换行符（'\n'）和可能的回车符（'\r'）
        stripped_line = line.rstrip('\n\r')
        # 将处理后的行添加到列表中
        all_loss.append(float(stripped_line))
fig = plt.figure()
ax = fig.add_subplot(111)
x = list(range(len(all_loss)))
ax.plot(x, all_loss, marker='o') # marker='o' 用于在每个数据点上绘制圆圈标记
ax.set_title('Training Loss')
ax.set_xlabel('Frequency')
ax.set_ylabel('Loss')
plt.show()