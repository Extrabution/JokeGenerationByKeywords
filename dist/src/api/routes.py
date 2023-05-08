import asyncio

from fastapi.templating import Jinja2Templates
from fastapi import Request, WebSocket
from configs.globals import task_queue, done_queue


templates = Jinja2Templates(directory="templates")


def register(app):
    @app.get("/")
    async def index(request: Request):
        return templates.TemplateResponse("index.html", {"request": request})

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket):
        await websocket.accept()
        while True:
            data = await websocket.receive_text()
            text, swear_check = data.split("|")
            if swear_check[1:] == "true":
                swear_flag = True
            else:
                swear_flag = False
            task_id = websocket.client.port
            task_queue.put_nowait((task_id, text, swear_flag))
            while True:
                id, generated_text = await done_queue.get()
                if id == task_id:
                    break
                else:
                    await done_queue.put((id, generated_text))
                    await asyncio.sleep(0.5)
            if generated_text is not None:
                await websocket.send_text(generated_text)

