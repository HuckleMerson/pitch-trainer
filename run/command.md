curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs
node -v
npm i -g pm2

apt install aria2

aria2c --enable-rpc --rpc-listen-all=true --rpc-allow-origin-all --dir=/tmp

sudo mount -t nfs -o ro,vers=4 185.141.218.201:/mnt/shared /mnt/shared


sudo add-apt-repository ppa:deadsnakes/ppa


mkdir /root/.config
mkdir /root/.config/Ultralytics

apt-get update && apt-get install -y nvidia-utils-535

apt install python3.12 -y python3.12-venv


sudo apt update
sudo apt install nfs-kernel-server

sudo apt update
sudo apt install nfs-common