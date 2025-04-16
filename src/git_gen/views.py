from contextlib import contextmanager

from rich.console import Console
from rich.status import Status
from rich.table import Table


class LinesStreamer:
    """Print multiple lines on console word by word by abusing status in rich package

    Example usage:
    ```python
    for texts in sentences:
        line_streamer.append(texts)
    ```
    """

    def __init__(self, status: Status, title: str) -> None:
        self.status = status
        self.title = title
        self.lines: list[str] = []

    def append(self, texts: list[str]):
        if len(texts) > len(self.lines):
            self.lines += [""] * (len(texts) - len(self.lines))
        for i, t in enumerate(texts):
            self.lines[i] = self.lines[i] + t
        self.render()

    def render(self):
        main_table = Table.grid()
        main_table.add_row(self.title)
        table = Table.grid(padding=(1, 4), pad_edge=True)
        for i, m in enumerate(self.lines):
            table.add_row(f"[i]{i}[/i]. {m}")
        main_table.add_row(table)
        self.status.update(main_table)


@contextmanager
def stream_lines(console: Console, title: str):
    """Print multiple lines on console word by word"""
    status = console.status("")
    status.start()
    try:
        line_streamer = LinesStreamer(status, title)
        line_streamer.render()
        yield line_streamer
    finally:
        status.stop()
