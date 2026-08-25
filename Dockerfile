FROM python:3.11-slim

RUN pip install --no-cache-dir \
    numpy==2.4.6 \
    opencv-python-headless==5.0.0.93 \
    pillow==12.3.0 \
    fastapi==0.141.1 \
    uvicorn==0.52.4 \
    python-multipart==0.0.32

WORKDIR /app
COPY solution.py grouper.py server.py ./
COPY static/ ./static/

# Default: the submission CLI (/input/images -> /output/predictions.csv).
# Demo server: docker run -p 8080:8080 <image> uvicorn server:app --host 0.0.0.0 --port 8080
CMD ["python", "solution.py"]
