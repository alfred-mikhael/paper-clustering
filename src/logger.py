import logging
from datetime import datetime
from typing import Optional, Any
import json

LOG_TO_STDOUT = True


class IngestionLogger():
    def __init__(self, logfile: str):
        self.logfile = logfile
        self.runtime = datetime.now().isoformat()

    def log_failure(self, e: Exception, stage: str, details: Optional[dict[str, Any]] = None):
        if LOG_TO_STDOUT:
            logging.exception(
                f"{type(e).__name__} occured in stage {stage} of pipeline. \n {str(e)} \n Details: {details}")

        with open(self.logfile, "a", encoding="utf-8") as f:
            json_error = {"type": e.__class__.__name__, "message": str(
                e), "details": details, "stage": stage, "arxiv_id": getattr(e, "arxiv_id", None)}
            print(json.dumps(json_error), file=f)
