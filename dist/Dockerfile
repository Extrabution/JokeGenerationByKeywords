FROM python
RUN apt-get install wget
WORKDIR /
COPY ../data data
COPY dist/src /dist/src



#
WORKDIR /dist
RUN pip install --no-cache-dir --upgrade -r ./src/requirements.txt


EXPOSE 8002:8002

WORKDIR /dist/src
CMD ["python3", "main.py"]