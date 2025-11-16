# 使用 conda
conda create -n aligned python=3.10
conda activate aligned
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 -c pytorch
pip install diffusers transformers sentencepiece protobuf==3.19.0

# huggingface配置，见 hg.sh

# 可能需要在 inference.py 文件内进行部分修改
python inference.py --model_path black-forest-labs/FLUX.1-dev --style_lambda 1.1

# 结果在 ./output/ 目录下