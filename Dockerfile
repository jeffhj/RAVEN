FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel

RUN conda install -y -c pytorch faiss-gpu=1.7.2 cudatoolkit=11.3

RUN pip install fairscale==0.4.13

RUN pip install transformers==4.18.0

RUN pip install rouge==1.0.1

RUN pip install tensorboard

RUN pip install wget

RUN pip install sentencepiece
