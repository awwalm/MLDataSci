import os

md = open("Images.md", "w")
path = os.getcwd()
dir_list = os.listdir(path)
for f in dir_list:
    if f.endswith(".png"):
        md.write(f"<img src={f}/>\n")
md.close()
