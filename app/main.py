
import uuid, traceback
from fastapi import FastAPI, Request, BackgroundTasks
from .config import settings
from .logger import logger
app = FastAPI()
@app.get("/")
def health():
    return {"status": "live"}
@app.post("/webhook/whatsapp")
async def whatsapp_webhook(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    trace_id = str(uuid.uuid4())
    logger.info("incoming", extra={"trace_id": trace_id})
    from .whatsapp.router import handle_whatsapp_event
    background_tasks.add_task(handle_whatsapp_event, data, trace_id)
    return {"ok": True}
