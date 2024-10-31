FROM python:3.12.3-slim

RUN apt-get update
RUN apt-get install nano
 
RUN mkdir wd
WORKDIR wd
COPY app/requirements.txt .
RUN pip3 install -r requirements.txt
 
COPY app/ ./
  
CMD [ "python3", "i2b2-usage-dashboard.py" ]
