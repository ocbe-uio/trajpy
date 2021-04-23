import trajpy.gui as gui
from ttkthemes import ThemedTk

root = ThemedTk(theme="clearlooks")
tj_gui = gui.trajpy_gui(root)
root.mainloop()