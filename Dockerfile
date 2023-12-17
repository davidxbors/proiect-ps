FROM python:3.11.7-bookworm

# set locale
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8     

# add a non-root user and switch to it
RUN useradd -m doa
USER doa 

WORKDIR /src

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "doa_comparison.py"]