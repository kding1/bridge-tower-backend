yes | conda create -n BT python=3.10 -c anaconda

eval "$(conda shell.bash hook)"

conda activate BT

git clone -b bt_dev https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-cpu.git
wget https://raw.githubusercontent.com/intel/intel-extension-for-pytorch/v2.0.100+cpu/scripts/compile_bundle.sh
bash compile_bundle.sh


git clone https://github.com/huggingface/transformers
cd transformers
pip install .
cd ..

git clone -b bt_dev https://github.com/intel/neural-compressor
cd neural-compressor
pip install .
cd ..


yes | conda install -c conda-forge pillow=9
yes | pip install opencv-python
yes | conda install -c conda-forge youtube-transcript-api
yes | conda install -c pytorch faiss-cpu
yes | pip install webvtt-py
yes | conda install -c conda-forge pytube
yes | conda install conda-pack -c conda-forge
yes | conda install -c anaconda chardet
pip install bottle
