import tkinter as tk
import tkinter.ttk as ttk

root = tk.Tk()
root.title("translation git")
root.geometry("800x600")
text = tk.Text(root)
button = tk.Button(root,textvariable="sdfjiijdf")
button.pack()
text.pack()
text.insert("1.0","sdufhudhfgr")
print(text.get("1.0",tk.END))

root.mainloop()
