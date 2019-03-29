FROM tensorflow/tensorflow:latest-py3
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

RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/community >> /etc/apk/repositories
RUN echo http://dl-cdn.alpinelinux.org/alpine/edge/testing >> /etc/apk/repositories
RUN apk add --update git cmake gcc make g++ py-pip python-dev openblas openblas-dev hdf5 hdf5-dev
RUN pip install pandas keras

CMD ["python", "main.py"]
