FROM python:3.9-slim-buster

WORKDIR /python-docker

export QT_DEBUG_PLUGINS=1
export DISPLAY=:1.0 

COPY requirements.txt requirements.txt
run export QT_DEBUG_PLUGINS=1
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 qt5-default  -y
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]