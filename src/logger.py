import logging
from datetime import datetime
from typing import Optional, Any
import json


class IngestionLogger:
    def __init__(self, logfile: str, skipfile: str, log_to_stdout: bool = False):
        self.logfile = logfile
        self.skipfile = skipfile
        self.runtime = datetime.now().isoformat()
        self.log_to_stdout = log_to_stdout

    def log_failure(
        self, e: Exception, stage: str, details: Optional[dict[str, Any]] = None
    ):
        if self.log_to_stdout:
            logging.exception(
                f"{type(e).__name__} occured in stage {stage} of pipeline at {self.runtime}. \n {str(e)} \n Details: {details}"
            )

        with open(self.logfile, "a", encoding="utf-8") as f:
            json_error = {
                "type": e.__class__.__name__,
                "message": str(e),
                "details": details,
                "stage": stage,
                "arxiv_id": getattr(e, "arxiv_id", None),
                "runtime": self.runtime,
            }
            print(json.dumps(json_error), file=f)

    def log_skip(self, arxiv_id: str, title: str, abstract: str, published: str):
        error_msg = f"Skipped {arxiv_id} at {self.runtime} published {published}, titled {title}. \n {abstract}"

        if self.log_to_stdout:
            logging.warning(error_msg)
        with open(self.skipfile, "a", encoding="utf-8") as f:
            print(error_msg, file=f)
