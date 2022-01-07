import sys
import os
arg = sys.argv[1]

os.system(f"cp docker_files/{arg}/Dockerfile ./")
os.system(f"sudo docker build --tag {arg} .")
os.system("rm Dockerfile")
