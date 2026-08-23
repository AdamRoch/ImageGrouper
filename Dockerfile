FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy==2.4.6 \
    opencv-python-headless==5.0.0.93 \
    pillow==12.3.0

WORKDIR /app
COPY solution.py grouper.py ./

CMD ["python", "solution.py"]
