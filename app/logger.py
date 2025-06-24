
import logging, sys
class TraceFormatter(logging.Formatter):
    def format(self, record):
        if not hasattr(record, "trace_id"):
            record.trace_id = "-"
        return super().format(record)
formatter = TraceFormatter("%(asctime)s [%(levelname)s] %(trace_id)s - %(message)s")
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)
logger = logging.getLogger("kongfood")
logger.addHandler(handler)
logger.setLevel(logging.INFO)
