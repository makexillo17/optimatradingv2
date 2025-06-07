# TODO: Modify this Procfile to fit your needs
web: gunicorn -k uvicorn.workers.UvicornWorker -w 1 --threads 8 --timeout 60 optimatrading.api.routes:app --bind 0.0.0.0:$PORT
