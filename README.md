# GNNBenchmark
## Ubuntu
https://blog.csdn.net/Wenyuanbo/article/details/126381967

## Linux kernel
### Check:
```
Uname -r
```
### Install: 
#### Mainline download:  https://kernel.ubuntu.com/mainline/
#### install : 
```
sudo dpkg -i *.deb
```
### Remove:
#### List all linux kernel: dpkg --list | grep linux-image
#### Remove: sudo apt purge linux-image-unsigned-5.4.0-190-generic
#### Fix: sudo apt --fix-broken install
#### Update: sudo update-grub

## VPN
### hitun.io, proxy software
### Set terminal proxy https://github.com/nityanandagohain/proxy_configuration
### Check the vpn in terminal: 
```
wget http://google.com
```
https://askubuntu.com/questions/831266/find-proxy-server-using-command-line
```
env | grep -i proxy
```
## Git install 
Sudo install git 

## CUDA Cudnn
If only run on CPU, no need to install cuda
CUDA driver: nvidia-smi
CUDA toolkit

## Docker Install 
https://inteldevzone.blog.csdn.net/article/details/121292624?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121292624-blog-138267662.235%5Ev43%5Epc_blog_bottom_relevance_base2&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-3-121292624-blog-138267662.235%5Ev43%5Epc_blog_bottom_relevance_base2&utm_relevant_index=6
```
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
```

### Check docker install
ping hub.docker.com

nslookup hub.docker.com

Sudo docker run hello-world

sudo tee /etc/docker/daemon.json <<-'EOF'
{
	"registry-mirrors":[
		"https://i1el1i0w.mirror.aliyuncs.com",
		"https://hub-mirror.c.163.com",
		"https://registry.aliyuncs.com",
		"https://registry.docker-cn.com",
		"https://docker.mirrors.ustc.edu.cn"
	]
}
EOF
### docker command ：
```
Docker image ls/ docker container ls
Docker ls -a
```


## Anaconda
### Instructions: 
https://blog.csdn.net/weixin_40964777/article/details/126308001
### Qinghua Source: 
https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/?C=M&O=A

## OpenVINO
### https://blog.csdn.net/sinat_28442665/article/details/120920760
```
Pip3 uninstall openvino
Pip3 install openvino=2024.02 –proxy=http://127.0.0.1:7890
```
## NPU check:
### Check npu:
```
lspci | grep -i npu
```
### Check npu driver:
```
lsmod | grep -i npu
```
## NPU driver
https://github.com/intel/linux-npu-driver/releases/tag/v1.5.1

## Intel NPU Acceleration library
### Install cmake: 
sudo apt install cmake
### Pip3 install: 
pip3 install intel-npu-acceleration-library
### Only Ubuntu22.04 is supported? 
https://community.intel.com/t5/Intel-Distribution-of-OpenVINO/NPU-not-detected-on-Ubuntu-with-OpenVINO-2024-1/m-p/1600697


## Pytorch
### https://pytorch.org/
### CPU install pytorch 2.30
```
pip3 install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
```
## PYG
Pytorch 2.30 install pyg
```
pip3 install torch_geometric
# Optional dependencies:
pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cpu.html
Pip3 install ogb
```
