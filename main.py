import os
import sys


path = './cleverhans/cleverhans_tutorials/'
if str(sys.argv[1]) == 'fgsm':
    os.system("CUDA_VISIBLE_DEVICES=\\'\\' python3 fgsm.py")
elif str(sys.argv[1]) == 'jsma':
    os.system("CUDA_VISIBLE_DEVICES=\\'\\' python3 " + path + "mnist_tutorial_jsma.py")
elif str(sys.argv[1]) == 'cw':
    os.system("CUDA_VISIBLE_DEVICES=\\'\\' python3 " + path + "mnist_tutorial_cw.py")
