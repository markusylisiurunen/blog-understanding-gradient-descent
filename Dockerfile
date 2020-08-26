FROM python:3

WORKDIR /var/app

RUN pip install numpy flask flask-cors

COPY model ./model
COPY server ./server

WORKDIR /var/app/server

ENV FLASK_APP mnist
ENV FLASK_DEBUG 0
ENV FLASK_ENV development

CMD [ "python", "main.py" ]
