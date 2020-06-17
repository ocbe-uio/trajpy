import tkinter as tk
from ttkthemes import ThemedTk
from functools import partial
import os
from PIL import ImageTk, Image
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import trajpy.trajpy as tj
import webbrowser

print("Tcl Version: {}".format(tk.Tcl().eval('info patchlevel')))
class trajpy_gui:

    def __init__(self, master):
        self.app = master
        self.init_window()
        self.app.resizable(False, False)
        self.title = tk.Label(self.app, text="TrajPy", font=("Arial Bold", 28))
        self._version = tk.Label(self.app, text="alpha", font=("Arial Bold", 12))
        self.entry = tk.Entry(self.app, width=50, highlightcolor='blue')
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.entry.insert(0, self.path)
        self.find_bt = tk.Button(self.app, text="Open", command=self.get_file)
        self.plot_bt = tk.Button(self.app, text="Plot", command=self.show_plot)
        self.button_select_all = tk.Checkbutton(self.app, text='Select all')
        self.button_select_all.configure(command=self.select_all)
        self.button_compute = tk.Button(self.app, text="Compute!", command=self.compute)
        self.results = tk.Entry(self.app, width=60, highlightcolor='blue')

        self.features = ['Anomalous Exponent', 'MSD Ratio', 'Fractal dimension',
                    'Anisotropy', 'Kurtosis', 'Straightness', 'Gaussianity', 'Confinement Prob.',
                    'Diffusivity', 'Efficiency']

        self.feats_ = []
        self.selected_features = []
        for feature in self.features:
            self.feats_.append(tk.Checkbutton(self.app, text=feature))

        self.placement()

    def init_window(self):
        self.app.title('TrajPy GUI alpha')
        self.app.geometry('600x600')

        self.menu = tk.Menu(self.app)
        self.app.config(menu=self.menu)


        self.file = tk.Menu(self.menu)

        self.file.add_command(label="Open...", command=self.get_file)
        self.file.add_command(label="Open directory...", command=self.get_file)
        self.file.add_command(label="Exit", command=self.client_exit)
        self.menu.add_cascade(label="File", menu=self.file)

        self.edit = tk.Menu(self.menu)
        self.edit.add_command(label="Undo")
        self.menu.add_cascade(label="Edit", menu=self.edit)

        self.help = tk.Menu(self.menu)
        self.help.add_command(label="About", command=self.About)
        self.menu.add_cascade(label="Help", menu=self.help)

    def client_exit(self):
        exit()

    def About(self):
        self.newWindow = tk.Toplevel(self.app)
        self.newWindow.resizable(False, False)
        self.img = Image.open(os.path.dirname(os.path.realpath(__file__))+'/logo.png')
        self.img = self.img.resize((300, 100), Image.BICUBIC)
        self.img = ImageTk.PhotoImage(self.img)    
        self._canvas = tk.Canvas(self.newWindow,width=300,height=100)
        self._canvas.pack()
        self._canvas.create_image(150, 50, image=self.img)
        
        #self._canvas.draw()
       
        #self.panel = tk.Label(self.newWindow, image=self.img)
        #self.panel.pack(side = "top", fill = "both", expand = "no")
        self.author = tk.Label(self.newWindow, text="Developed by Maurício Moreira-Soares")
        self.link = tk.Label(self.newWindow, text="phydev.github.io", fg='blue')
        self.email =tk.Label(self.newWindow, text="mms@uc.pt")
        self.link.bind("<Button-1>", lambda e: webbrowser.open_new("https://phydev.github.io"))
        self.author.pack()
        self.link.pack()
        self.email.pack()


    def placement(self):
        self.title.place(x=250, y=10)
        self._version.place(x=300,y=50)
        self.entry.place(x=80, y=100)
        self.find_bt.place(x=380, y=130)
        self.plot_bt.place(x=440, y=130)
        self.button_select_all.place(x=20, y=230 + (len(self.feats_) + 1) * 20)
        self.button_compute.place(x=20, y=230 + (len(self.feats_) + 3) * 20)
        self.results.place(x=20, y=230 + (len(self.feats_) + 5) * 20)
        for n, button in enumerate(self.feats_):
            self.feats_[n].configure(command=partial(self.select_feat, self.feats_[n]))
            self.feats_[n].place(x=20, y=220 + n * 20)

    def open(self):
        self.r = tj.Trajectory(self.path, skip_header=1, delimiter=',')

    def compute(self):
        results = self.r.compute_features
        self.results.insert(0,results)

    def get_file(self):
        filename = tk.filedialog.askopenfilename(parent=self.app,
                                                 initialdir=self.path,
                                                 title='Please select a file')
        self.path = filename
        self.entry.delete(0, 100)
        self.entry.insert(0, filename)

    def show_plot(self):
        self.open()
        self._fig = Figure(figsize=(3, 3), dpi=100)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self.app)
        self._fig.add_subplot(111).plot(self.r._t, self.r._r, ls='-.')
        self._canvas.draw()
        self._canvas.get_tk_widget().place(x=200, y=200)

    def select_feat(self, button):
        if any(button.cget('text') in feature for feature in self.selected_features):
            self.selected_features.remove(button.cget('text'))
            button.deselect()
        else:
            self.selected_features.append(button.cget('text'))
            button.select()

    def select_all(self):
        self.selected_features = []
        for n, button in enumerate(self.feats_):
            if self.button_select_all.cget('text') == 'Select all':
                button.select()
                button.configure(state='disabled')
                self.selected_features.append(button.cget('text'))

                update_text = 'Deselect all'
            else:
                button.configure(state='active')
                button.deselect()
                update_text = 'Select all'
        self.button_select_all.configure(text=update_text)



# write ticking marks to select features to be computed

if __name__ == '__main__':
    root = ThemedTk(theme="clearlooks")
    tj_Gui = trajpy_gui(root)
    root.mainloop()
