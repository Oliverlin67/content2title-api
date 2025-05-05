FROM python:3.11-slim-buster

# setup worker location
WORKDIR /app

# copy files
COPY requirements.txt .
COPY main.py .
COPY model_save model_save

# install python packages
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

# startup command
CMD ["gunicorn", "-w", "1", "--threads", "4", "-b", "0.0.0.0:8080", "main:app"]