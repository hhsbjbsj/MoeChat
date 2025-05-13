import re
 
text = "这是[example1]测试[example2]字符串[example3]"
# matches = re.findall(r'\[(.*?)\]', text)
matches = re.findall(r'\[(.*?)\]', text)
print(matches)  # 输出: ['example1', 'example2', 'example3']

