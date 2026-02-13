import tkinter as tk
from queue import Empty

FONT_FAMILY = "Helvetica"
FONT_SIZE = 36  # increase to 42–48 for demo rooms

class LLM_UI:
    def __init__(self, llm_queue):
        self.llm_queue = llm_queue

        self.root = tk.Tk()
        self.root.title("SAMI says")

        self.root.configure(bg="black")
        self.root.geometry("900x300")

        self.label = tk.Label(
            self.root,
            text="",
            font=(FONT_FAMILY, FONT_SIZE, "bold"),
            fg="white",
            bg="black",
            wraplength=850,
            justify="center"
        )
        self.label.pack(expand=True)

        self.poll_queue()

    def poll_queue(self):
        try:
            while True:
                text = self.llm_queue.get_nowait()
                self.label.config(text=text)
        except Empty:
            pass

        self.root.after(100, self.poll_queue)

    def run(self):
        self.root.mainloop()
