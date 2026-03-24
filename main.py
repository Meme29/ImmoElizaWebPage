from fastapi import FastAPI
from predict import predictor
from preprocessing import preprocessor
from fastapi.responses import HTMLResponse


app = FastAPI()

app.include_router(preprocessor.router, prefix="/preprocess", tags=["Preprocessor"])
app.include_router(predictor.router, prefix="/predict", tags=["Predictor"])


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("./pages/homepage.html", encoding="utf-8") as f:
        return f.read()