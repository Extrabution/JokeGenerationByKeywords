from __future__ import annotations
from fastapi import FastAPI

import uvicorn
from multiprocessing import cpu_count, freeze_support
from api.routes import register
from services.joke_generator import start_workers
import asyncio

app = FastAPI()
register(app)
workers:list

@app.on_event('startup')
async def initial_task():
    global workers
    workers = await start_workers(2)


def main():
    try:
        freeze_support()
        num_workers = int(cpu_count() * 0.75)
        start_server(num_workers=1, reload=False, host='0.0.0.0')
    except Exception as e:
        print("Something bad happened!", e)
        for worker in workers:
            worker.cancel()

        asyncio.gather(*worker, return_exceptions=True)
        print("Workers stopped")

def start_server(host='127.0.0.1', port=8002, num_workers=4, loop='asyncio', reload=False):
    uvicorn.run('main:app', host=host,
                port=port,
                workers=num_workers,
                loop=loop,
                reload=reload)

if __name__ == '__main__':
    main()
