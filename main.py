from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.wsgi import WSGIMiddleware
from fastapi.templating import Jinja2Templates
import uvicorn
from dataclasses import dataclass
from contextlib import asynccontextmanager
from typing import Annotated
import joblib

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

ml_models = {}


@app.on_event("startup")
async def startup_event():
    scaler_for_diabetes = joblib.load('./data/diabetes_standard_scaler_joblib')
    diabetes_model = joblib.load('./data/diabetes_joblib')
    lung_cancer = joblib.load("./data/lung_cancer_main_joblib")
    ml_models['diabetes_model'] = diabetes_model
    ml_models['scaler_for_diabetes'] = scaler_for_diabetes
    ml_models['lung_cancer'] = lung_cancer
    print("Models is Loading")


@app.get("/items/")
async def end_event():
    return ml_models.clear()


@ app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@ app.get("/diabetes", response_class=HTMLResponse)
async def get_diabetes(request: Request):
    return templates.TemplateResponse("diabetes.html", {'request': request})


@ app.post("/diabetes", response_class=HTMLResponse)
async def get_diabetes(request: Request,
                       Pregenesy: Annotated[int, Form()],
                       Glucose: Annotated[int, Form()],
                       BloodPressure: Annotated[int, Form()],
                       SkinThickness: Annotated[float, Form()],
                       Insuline: Annotated[float, Form()],
                       BMI: Annotated[float, Form()],
                       DiabetesPedigreeFunction: Annotated[float, Form()],
                       Age: Annotated[int, Form()],
                       ):
    print(Pregenesy, Glucose, BloodPressure, SkinThickness,
          Insuline, BMI, DiabetesPedigreeFunction, Age)
    nums = ml_models["scaler_for_diabetes"].transform(
        [[Pregenesy, Glucose, BloodPressure, SkinThickness, Insuline, BMI, DiabetesPedigreeFunction, Age]])
    result = ml_models["diabetes_model"].predict(nums)
    print(f"So the result is : {result}")
    diabetes = ['No', 'Yes']
    return templates.TemplateResponse("diabetes.html", {'request': request, 'result': diabetes[result[0]]})


@ app.get("/lungcancer", response_class=HTMLResponse)
async def get_lung_cancer(request: Request):
    return templates.TemplateResponse("lung-cancer.html", {'request': request})


@dataclass
class LungCancerForm:
    GENDER: Annotated[int, Form()]
    AGE: Annotated[int, Form()]
    SMOKING: Annotated[int, Form()]
    YELLOW_FINGERS: Annotated[int, Form()]
    ANXIETY: Annotated[int, Form()]
    PEER_PRESSURE: Annotated[int, Form()]
    CHRONIC_DISEASE: Annotated[int, Form()]
    FATIGUE: Annotated[int, Form()]
    ALLERGY: Annotated[int, Form()]
    WHEEZING: Annotated[int, Form()]
    ALCOHOL_CONSUMING: Annotated[int, Form()]
    COUGHING: Annotated[int, Form()]
    SHORTNESS_OF_BREATH: Annotated[int, Form()]
    SWALLOWING_DIFFICULTY: Annotated[int, Form()]
    CHEST_PAIN: Annotated[int, Form()]


@ app.post("/lungcancer", response_class=HTMLResponse)
async def get_lung_cancer(request: Request, GENDER: Annotated[int, Form()], AGE: Annotated[int, Form()], SMOKING: Annotated[int, Form()],
                          YELLOW_FINGERS: Annotated[int, Form()], ANXIETY: Annotated[int, Form()], PEER_PRESSURE: Annotated[int, Form()],
                          CHRONIC_DISEASE: Annotated[int, Form()], FATIGUE: Annotated[int, Form()], ALLERGY: Annotated[int, Form()],
                          WHEEZING: Annotated[int, Form()], ALCOHOL_CONSUMING: Annotated[int, Form()], COUGHING: Annotated[int, Form()],
                          SHORTNESS_OF_BREATH: Annotated[int, Form()], SWALLOWING_DIFFICULTY: Annotated[int, Form()], CHEST_PAIN: Annotated[int, Form()],
                          ):
    nums = [[GENDER, AGE, SMOKING, YELLOW_FINGERS, ANXIETY, PEER_PRESSURE, CHRONIC_DISEASE, FATIGUE, ALLERGY,
             WHEEZING, ALCOHOL_CONSUMING, COUGHING, SHORTNESS_OF_BREATH, SWALLOWING_DIFFICULTY, CHEST_PAIN]]
    result = ml_models["lung_cancer"].predict(nums)
    print(f"So the result is : {result}")
    diabetes = ['No', 'Yes']
    return templates.TemplateResponse("lung-cancer.html", {'request': request, "result": diabetes[result[0]]})


if __name__ == "__main__":
    uvicorn.run("main:app", host='127.0.0.1', port=8000, reload=True)
