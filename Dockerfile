FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

LABEL taost=taost

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

ENV CUDA_VISIBLE_DEVICES=7

ENV PORT=8800

EXPOSE 8800

WORKDIR ./DEQ-Sequence

# CMD bash penn_deq_transformer.sh train --cuda --multi_gpu --f_solver broyden --f_thres 30 --b_thres 40

# CMD python train_transformer.py --name taostPennTreebank --work_dir . --cuda --data ./data/penn/ --dataset ptb

CMD bash penn_deq_transformer.sh train --cuda --f_solver broyden --f_thres 30 --b_thres 40

# CMD python -c 'import torch; print(torch.version.cuda, torch.cuda.is_available())'

