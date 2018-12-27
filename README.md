# 一个神经网络对抗样本生成工具
## 安装方法
```python
git clone https://github.com/wazxser/DNN_testing.git
```
然后需要安装cleverhans库
```python
cd DNN_testing

rm cleverhans

git clone https://github.com/tensorflow/cleverhans

pip3 install -e ./cleverhans
```
## 运行方式
```angular2
python3 main [option]
```
```python
option:   fgsm
          jsma
          cw
```