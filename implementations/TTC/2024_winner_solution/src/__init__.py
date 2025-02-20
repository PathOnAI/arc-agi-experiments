import logging
import os
import typing as T

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("PRINT_LOGS") or os.environ["PRINT_LOGS"] == "0":
    PRINT_LOGS = False
else:
    PRINT_LOGS = True


class LogfireDummy:
    def __init__(self):
        self.logger = logging.getLogger("LogfireDummy")
        self.logger.setLevel(logging.DEBUG)
        handler = logging.FileHandler("logs.txt")
        formatter = logging.Formatter(
            "%(asctime)s EST: %(message)s", datefmt="%Y-%m-%d %I:%M:%S %p"
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def debug(
        self,
        msg_template: str,
        /,
        **attributes: T.Any,
    ) -> None:
        try:
            s = msg_template
            if attributes:
                s = f"{s}\n**{attributes}**\n"
            if PRINT_LOGS:
                print(f"LOGGER: {s}")
            else:
                if "KAGGLE" not in os.environ:
                    self.logger.debug(s)
                else:
                    # make sure no non-sdk logs are recorded
                    if "anthropic" in s or "Transform" in s or "limit" in s:
                        self.logger.debug(s)
        except Exception:
            pass


if os.getenv("LOGFIRE_TOKEN"):
    import logfire

    logfire.configure(inspect_arguments=True)
else:
    logfire = LogfireDummy()


if not os.getenv("PLOT") or os.environ["PLOT"] == "0":
    PLOT = False
else:
    PLOT = True

if os.environ.get("USE_GRID_URL", "1") == "0":
    USE_GRID_URL = False
else:
    USE_GRID_URL = True
