FROM python:3.7.2-alpine3.9
ADD . /app
WORKDIR /app
EXPOSE 3000
RUN apk add --update python3 python3-dev gfortran py-pip build-base g++ gfortran file binutils \
                     musl-dev openblas-dev libstdc++ openblas
RUN pip install numpy
RUN pip install np
RUN pip install sklearn
RUN pip install nltk
RUN python setup.py

CMD ["python", "main.py"]
