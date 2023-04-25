yes | conda create -n BT python=3.8 -c anaconda

eval "$(conda shell.bash hook)"

conda activate BT
git clone https://github.com/huggingface/transformers
cd transformers
pip install .
yes | conda install -c conda-forge huggingface_hub
yes | conda install pytorch  cpuonly -c pytorch
yes | pip install intel_extension_for_pytorch -f https://developer.intel.com/ipex-whl-stable-cpu
#yes | conda install -c intel intel-extension-for-pytorch
yes | conda install -c anaconda requests
yes | conda install -c conda-forge pillow=9
yes | pip install opencv-python
yes | conda install -c conda-forge youtube-transcript-api
yes | conda install -c pytorch faiss-cpu
yes | pip install webvtt-py
yes | conda install -c conda-forge pytube
yes | conda install conda-pack -c conda-forge
yes | conda install -c anaconda chardet
pip install bottle
