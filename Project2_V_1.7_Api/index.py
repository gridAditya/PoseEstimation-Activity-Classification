from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes.pose import pose

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(pose)