FROM python:3.10

ENV CUDA_VISIBLE_DEVICES=5

ENV PORT=8800

EXPOSE 8800

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

WORKDIR /DEQ-Sequence

CMD bash wt103_deq_transformer.sh train --f_thres 30 --eval --load pretrained_wt103_deqtrans_v3.pkl --mem_len 300 --pretrain_step 0 --name pretrained

