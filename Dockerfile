FROM python:3.7-slim-buster
COPY . /name_entity_recognition
WORKDIR /name_entity_recognition
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install -e .
CMD ["python","app.py"]