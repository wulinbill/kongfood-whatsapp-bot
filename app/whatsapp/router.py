
from ..config import settings
from .twilio_adapter import send_message as twilio_send
from .dialog360_adapter import send_message as dialog_send
from ..oco_core.seed_parser import parse
from ..oco_core.output_director import reply
from ..logger import logger
def handle_whatsapp_event(payload, trace_id):
    text = payload.get('Body') or ''
    co = parse(text)
    answer = reply(co)
    phone = payload.get('From')
    send = twilio_send if settings.WHATSAPP_VENDOR=='twilio' else dialog_send
    send(phone, answer)
    logger.info('replied', extra={'trace_id': trace_id})
