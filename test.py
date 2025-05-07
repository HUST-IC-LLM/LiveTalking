import subprocess

# 创建一个管道：grep通过ps获取的进程列表
ps = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
grep = subprocess.Popen(['grep', 'ollama'], stdin=ps.stdout, stdout=subprocess.PIPE)
ps.stdout.close() # 允许ps释放资源
output = grep.communicate()[0]
print(output.decode())