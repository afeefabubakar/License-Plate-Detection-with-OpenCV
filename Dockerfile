FROM python:3.8-slim

ENV PYTHONUNBUFFERED=1
RUN apt update \
    && apt -y install tesseract-ocr \
    && apt -y install ffmpeg libsm6 libxext6

ADD anprdriver.py .
ADD anprclass.py .
ADD requirements.txt .
ADD /malaysian /malaysian
ADD /overseas /overseas

RUN pip install -r requirements.txt

ENTRYPOINT ["/bin/sh"]