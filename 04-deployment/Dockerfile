FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

RUN pip install -U pip
RUN pip install pipenv

COPY [ "Pipfile", "Pipfile.lock", "./" ]

RUN pipenv install --system --deploy

COPY [ "homework/starter.py", "starter.py"]

ENTRYPOINT [ "python", "starter.py" ]

# Build the image with:
#    `docker build -t ride-duration-prediction-batch:v1 .`
# Run the container with:
#    `docker run -it --rm ride-duration-prediction-batch:v1 --taxi_type yellow --year 2023 --month 5`