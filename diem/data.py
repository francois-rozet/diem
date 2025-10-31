r"""Data helpers"""

from queue import Queue
from threading import Thread
from typing import *


class prefetch(Thread):
    def __init__(self, iterable: Iterable, buffer: int = 1):
        super().__init__(daemon=True)

        self.queue = Queue(buffer)
        self.iterable = iterable
        self.end = object()
        self.start()

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> Any:
        item = self.queue.get()
        if item is self.end:
            raise StopIteration
        return item

    def run(self):
        for item in self.iterable:
            self.queue.put(item)
        self.queue.put(self.end)
