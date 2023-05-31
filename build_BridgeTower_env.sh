yes | conda create -n BT python=3.10 -c anaconda

eval "$(conda shell.bash hook)"

conda activate BT


python -m pip install cmake
python -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

python -m pip install -r requirements.txt
python -m pip install --force-reinstall *.whl


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
