from datetime import datetime
from math import floor
import os


def seconds_to_str(seconds: float) -> str:
    """Convert seconds to a human readable string"""
    seconds_int = floor(seconds)
    hours = seconds_int // 3600
    minutes = (seconds_int % 3600) // 60

    # seconds and milliseconds
    milliseconds = int((seconds % 1) * 1000)
    seconds_int = seconds_int % 60

    # format
    if hours > 0:
        return f"{hours}h {minutes}m {seconds_int}s {milliseconds}ms"
    elif minutes > 0:
        return f"{minutes}m {seconds_int}s {milliseconds}ms"
    elif seconds > 0:
        return f"{seconds_int}s {milliseconds}ms"
    else:
        return f"{milliseconds}ms"


class Timer:
    """
    Context manager to time a block of code

    You can set the environment variable TIMER_STATUS to "loud" to
    enable the timer or not setting this at all. Otherwise, it will be silent.
    """

    def __init__(
        self,
    ):
        self.silence = os.environ.get("TIMER_STATUS", "loud")
        self.start = datetime.utcnow()
        self.last = datetime.utcnow()

    def __call__(self, task: str):
        if str(self.silence) != "loud":
            return self
        end = datetime.utcnow()
        total = seconds_to_str((end - self.start).total_seconds())
        span = seconds_to_str((end - self.last).total_seconds())
        print(
            f"🏁 >> {task}: {span} (total: {total}): (Start: {self.start.strftime('%M:%S.%f')} -> End: {end.strftime('%M:%S.%f')})"
        )
        self.last = end
        return self
