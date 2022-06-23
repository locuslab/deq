FROM python:3.10

ENV CUDA_VISIBLE_DEVICES=5

ENV PORT=8800

EXPOSE 8800

WORKDIR /DEQ-Sequence

COPY ./requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD echo "testing 123"

