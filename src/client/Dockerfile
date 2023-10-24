FROM tiangolo/uvicorn-gunicorn:python3.8
RUN mkdir -p /app
WORKDIR /app 

ENV TZ Asia/Seoul
ENV LC_ALL C.UTF-8

COPY ./src/client/requirements.txt /requirements.txt

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /requirements.txt

COPY ./src/client/app/ /app/

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:80", "--log-level", "debug", "app:app"]