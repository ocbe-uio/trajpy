import tkinter as tk
from ttkthemes import ThemedTk
from functools import partial
import os
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns


class trajpy_gui:

    def __init__(self, master):
        self.app = master
        self.app.title('TrajPy')
        self.app.geometry('600x600')
        self.title = tk.Label(self.app, text="TrajPy", font=("Arial Bold", 28))
        self.entry = tk.Entry(self.app, width=50, highlightcolor='blue')
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.entry.insert(0, self.path)
        self.canvas = FigureCanvasTkAgg(fig, master=self.app)  # A tk.DrawingArea.
        self.find_bt = tk.Button(self.app, text="Find", command=self.get_file)
        self.plot_bt = tk.Button(self.app, text="Plot", command=self.show_plot)
        self.button_select_all = tk.Checkbutton(self.app, text='Select all')
        self.button_select_all.configure(command=self.select_all)


        self.features = ['Anomalous Exponent', 'MSD Ratio', 'Fractal dimension',
                    'Anisotropy', 'Kurtosis', 'Straightness', 'Gaussianity', 'Confinement Prob.',
                    'Diffusivity', 'Efficiency']

        self.feats_ = []
        self.selected_features = []
        for feature in self.features:
            self.feats_.append(tk.Checkbutton(self.app, text=feature))

        self.placement()

    def placement(self):
        self.title.place(relx=0.4, rely=0)
        self.entry.place(relx=0.1, rely=0.2)
        self.find_bt.place(relx=0.1, rely=0.3)
        self.plot_bt.place(relx=0.3, rely=0.3)
        self.button_select_all.place(x=20, y=230 + (len(self.feats_) + 1) * 20)

        for n, button in enumerate(self.feats_):
            self.feats_[n].configure(command=partial(self.select_feat, self.feats_[n]))
            self.feats_[n].place(x=20, y=220 + n * 20)


    # actions
    def get_file(self):
        filename = tk.filedialog.askopenfilename(parent=self.app,
                                                 initialdir=self.path,
                                                 title='Please select a file')
        self.entry.delete(0, 100)
        self.entry.insert(0, filename)

    def show_plot(self):
        self.canvas.draw()
        self.canvas.get_tk_widget().place(relx=0.3, rely=0.6)

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
    fig = Figure(figsize=(3, 2), dpi=100)
    t = np.arange(0, 3, .01)
    fig.add_subplot(111).plot(t, 2 * np.sin(2 * np.pi * t), ls='-.')
    root = ThemedTk(theme="randiance")
    tj_Gui = trajpy_gui(root)
    root.mainloop()
