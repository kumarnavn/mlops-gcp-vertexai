FROM python:3.7

WORKDIR /root

COPY requirements.txt ./
COPY trainer ./trainer

RUN pip3 install --requirement requirements.txt

WORKDIR /root/trainer

CMD ["python3", "task.py"]
