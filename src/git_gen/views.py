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

    def build_renderable(self):
        main_table = Table.grid()
        main_table.add_row(self.title)
        table = Table.grid(padding=(0, 4), pad_edge=True)
        for i, m in enumerate(self.lines):
            table.add_row(f"[i]{i}[/i]. {m}")
        main_table.add_row(table)
        return main_table

    def render(self):
        widget = self.build_renderable()
        self.status.update(widget)


@contextmanager
def stream_lines(console: Console, title: str):
    """Print multiple lines on console word by word"""
    status = console.status("")
    status.start()
    line_streamer = LinesStreamer(status, title)
    try:
        line_streamer.render()
        yield line_streamer
    finally:
        console.print(line_streamer.build_renderable())
        status.stop()
