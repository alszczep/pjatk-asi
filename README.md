# pjatk-asi

## Setting up the local environment

Recommended python version is 3.12.
Create the venv - e.g. with `python -m venv venv` or using PyCharm.
To install the dependencies run in each app's directory:

`[path/to/venv]/Scripts/pip.exe install -r [path/to/project]requirements.txt`

## Dataset

https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam?resource=download

## Downloading the dataset

Download the dataset manually and extract into `data/01_raw`.

## .env

.env files should be created in the `backend` and `frontend` folders or provided by the docker container.

Environment variables for the api:
- BLOB_STORAGE_CONNECTION_STRING
- API_PORT

Environment variables for the frontend:
- FRONTEND_PORT
- API_BASE_URL

## Running the app locally

Apps can be run locally without docker using:

`[path/to/venv]/Scripts/python.exe run [api or frontend]main.py`

## Running using docker compose 

Apps can be run using docker compose by running:
```
docker-compose up --build frontend
docker-compose up --build api
```

## Training the model
To train the model, run the following command in the asi directory:

`[path/to/venv]/Scripts/kedro.exe run`
