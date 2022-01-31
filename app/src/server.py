import os
from tempfile import SpooledTemporaryFile

import redis
from fastapi import FastAPI, UploadFile, BackgroundTasks, Form
from fastapi.responses import HTMLResponse, RedirectResponse, FileResponse
from starlette.concurrency import run_in_threadpool

from model_generator import ModelGenerator

app = FastAPI(debug=True)
DATABASE = redis.Redis(host='redis', port=6379, decode_responses=True)
DATABASE.set('model_num', 1000)


async def get_predict(train_file: SpooledTemporaryFile, test_file: SpooledTemporaryFile, target_feature: str,
                      model_num: int):
    model_generator = ModelGenerator(DATABASE, train_file, test_file, target_feature, model_num)
    await run_in_threadpool(model_generator.predict)


@app.get('/model/{model_num}', response_class=HTMLResponse)
def get_model_result(model_num: int):
    """"""
    model_info = DATABASE.get(str(model_num))
    if model_info is None:
        return 'Model not found!'
    elif model_info == 'In Progress...':
        return 'In Progress...'
    if os.path.isfile(model_info):
        path = model_info
        return f"""
        <body>
            <p><a href="/download/?path={path}">{path}</a></p>
        </body>
        """


@app.get('/download/')
def download_csv(path: str):
    return FileResponse(path)


@app.get('/')
def get_status():
    return 'OK'


@app.get('/upload_file', response_class=HTMLResponse)
def upload_file():
    """Простая форма для загрузки двух датасэтов и целевой переменной"""
    return """
    <form method="post" enctype="multipart/form-data">
        <div>
            <label for="file">Выберете файл с тренировочными данными</label>
            <input type="file" id="file" name="train" accept=".csv">
        </div>
        <div>
            <label for="file">Выберете файл с тестовыми данными</label>
            <input type="file" id="file" name="test" accept=".csv">
        </div>
        <div>
            <label for="file">Введите название целевой переменной</label>
            <input type="text" name="target_feature">
        </div>
        <div>
            <button>Submit</button>
        </div>
    </form>
    """


@app.post('/upload_file')
def upload_file(train: UploadFile, test: UploadFile, background_tasks: BackgroundTasks,
                target_feature: str = Form('target_feature')):
    """Загрузка файла и добавление в фон таски обучения"""
    model_num = int(DATABASE.get('model_num'))
    model_num += 1
    background_tasks.add_task(get_predict, train.file, test.file, target_feature, model_num)
    DATABASE.set(str(model_num), 'In Progress...')
    DATABASE.set('model_num', model_num)
    return RedirectResponse(status_code=303, url=f'/model/{model_num}')
