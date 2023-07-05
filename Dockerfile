FROM tensorflow/tensorflow:latest-gpu
WORKDIR ./DeepLearningModel
COPY . .
RUN apt-get update

RUN pip install matplotlib
RUN pip install scikit-learn
RUN pip install numpy

ENTRYPOINT ["python3", "main.py", "test.py", "sample.py"]