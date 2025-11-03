1. Start clash
cd /home/liuchi/clash
./clash -d .

2. Setup the scripts. Add following to your scripts.

export all_proxy="socks5://127.0.0.1:7890"
export ALL_PROXY="socks5://127.0.0.1:7890"
export https_proxy="http://127.0.0.1:7890"
export http_proxy="http://127.0.0.1:7890"

python main.py
