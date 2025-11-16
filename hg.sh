huggingface-cli whoami

# huggingface 登录
# 需要设置一个可行的token
    # https://huggingface.co/settings/tokens
huggingface-cli login

# 后台安装Flux，这里安装到 ./Tdir/ 目录内
nohup python inference.py --model_path black-forest-labs/FLUX.1-dev --style_lambda 1.1 > output.log 2>&1 &

# 查看是否在安装
du -sh Tdir
# 或
tail -50f output.log