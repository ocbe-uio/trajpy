from tkinter import filedialog
import tkinter as tk
from ttkthemes import ThemedTk
from functools import partial
import os
from PIL import ImageTk, Image
import numpy as np
import trajpy.trajpy as tj
import webbrowser
import trajpy
from typing import List, Union, Dict, Optional

print("Tcl Version: {}".format(tk.Tcl().eval('info patchlevel')))
print("TrajPy Version: {}".format(trajpy.__version__))
print("Loading TrajPy GUI...")

class trajpy_gui:

    def __init__(self, master: ThemedTk) -> None:
        self.app = master
        self.init_window()
        self.app.resizable(False, False)
        self.title = tk.Label(self.app, text="TrajPy", font=("Arial Bold", 28))
        self._version = tk.Label(self.app, text="{}".format(trajpy.__version__), font=("Arial Bold", 12))
        self.entry = tk.Entry(self.app, width=50, highlightcolor='blue')
        self.path = os.path.dirname(os.path.realpath(__file__))
        self.entry.insert(0, self.path)
        self.find_bt = tk.Button(self.app, text="Open file...", command=partial(self.get_file, 'file'))
        self.find_dir_bt = tk.Button(self.app, text="Open directory...", command=partial(self.get_file, 'dir'))
        self.button_select_all = tk.Checkbutton(self.app, text='Select all')
        self.button_select_all.configure(command=self.select_all)
        self.button_compute = tk.Button(self.app, text="Compute!", command=self.compute_selected)
        self.results = tk.Entry(self.app, width=60, highlightcolor='blue')

        self.features = ['Anomalous Exponent', 'MSD Ratio', 'Fractal dimension',
                         'Anisotropy & Kurtosis', 'Straightness', 'Efficiency',
                         'Gaussianity', 'Diffusivity', 'Confinement Prob.']

        self.feats_ = {}
        self.selected_features = []
        self.data = {}
        self.files = []
        self.trajectory_list = []

        for feature in self.features:
            self.feats_[feature] = tk.Checkbutton(self.app, text=feature)

        self.placement()

    def init_window(self) -> None:
        self.app.title('TrajPy GUI')
        self.app.geometry('600x600')

        self.menu = tk.Menu(self.app)
        self.app.config(menu=self.menu)

        self.file = tk.Menu(self.menu)

        self.file.add_command(label="Open file...", command=partial(self.get_file, 'file'))
        self.file.add_command(label="Open directory...", command=partial(self.get_file, 'dir'))
        self.file.add_command(label="Save as...", command=self.save_file)
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

    def About(self) -> None:
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
        self.email =tk.Label(self.newWindow, text="trajpy@protonmail.ch")
        self.link.bind("<Button-1>", lambda e: webbrowser.open_new("https://phydev.github.io"))
        self.author.pack()
        self.link.pack()
        self.email.pack()


    def placement(self) -> None:
        self.title.place(x=250, y=10)
        self._version.place(x=300,y=50)
        self.entry.place(x=80, y=100)
        self.find_bt.place(x=250, y=130)
        self.find_dir_bt.place(x=320, y=130)
        self.button_select_all.place(x=20, y=230 + (len(self.feats_) + 1) * 20)
        self.button_compute.place(x=20, y=230 + (len(self.feats_) + 3) * 20)
        self.results.place(x=20, y=230 + (len(self.feats_) + 5) * 20)
        for n, button in enumerate(self.feats_):
            self.feats_[button].configure(command=partial(self.select_feat, self.feats_[button]))
            self.feats_[button].place(x=20, y=220 + n * 20)

    def open(self, kind: str) -> None:

        if kind=="file":
            self.trajectory_list.append(tj.Trajectory(self.path, skip_header=1, delimiter=','))

        elif kind=="dir":
            
            # open all files and save trajectories in a list
            self.trajectory_list = []
            for file in self.files:
                print(self.path, file)
                self.trajectory_list.append(tj.Trajectory(self.path + "/"+ file, skip_header=1, delimiter=','))
                

    def compute_selected(self) -> None:
        '''
            compute the selected features
        '''
        
        if self.kind=="dir":
            f = filedialog.asksaveasfile(mode='w', defaultextension=".csv")

            if f is None:
                print("No file selected!")
                return

        self.data = {}

        for n, trajectory in enumerate(self.trajectory_list):
            
            self.r = trajectory

            self.data[n] = {}
            
            if any('Anomalous' in feature for feature in self.selected_features):
                self.r.msd_ea = self.r.msd_ensemble_averaged_(self.r._r)
                self.r.anomalous_exponent = self.r.anomalous_exponent_(self.r.msd_ea, self.r._t)
                self.data[n]['alpha'] = self.r.anomalous_exponent

            if any('MSD' in feature for feature in self.selected_features):
                self.r.msd_ta = self.r.msd_time_averaged_(self.r._r, np.arange(len(self.r._r)))
                self.r.msd_ratio = self.r.msd_ratio_(self.r.msd_ta, n1=2, n2=10)
                self.data[n]['msd_ratio'] = self.r.msd_ratio

            if any('Fractal' in feature for feature in self.selected_features):
                self.r.fractal_dimension, self.r._r0 = self.r.fractal_dimension_(self.r._r)
                self.data[n]['df'] = self.r.fractal_dimension

            if any(item in feature for feature in self.selected_features for item in ['Kurtosis', 'Anisotropy']):
                gyration_radius_dict = self.r.gyration_radius_(self.r._r)
                self.r.gyration_radius = gyration_radius_dict['gyration tensor']
                self.r.eigenvalues = gyration_radius_dict['eigenvalues'] 
                self.r.eigenvectors = gyration_radius_dict['eigenvectors']
                #self.r.eigenvalues[::-1].sort()  # the eigenvalues must be in the descending order
                #self.r._idx = self.r.eigenvalues.argsort()[::-1]  # getting the position of the principal eigenvector
                self.r.kurtosis = self.r.kurtosis_(self.r._r, self.r.eigenvectors[:,0])
                self.r.anisotropy = self.r.anisotropy_(self.r.eigenvalues)
                self.data[n]['anisotropy'] = self.r.anisotropy
                self.data[n]['kurtosis'] = self.r.kurtosis

            if any('Gaussianity' in feature for feature in self.selected_features):
                self.r.gaussianity = self.r.gaussianity_(self.r._r)
                self.data[n]['gaussianity'] =  self.r.gaussianity

            if any('Straightness' in feature for feature in self.selected_features):
                self.r.straightness = self.r.straightness_(self.r._r)
                self.data[n]['straightness'] = self.r.straightness

            if any('Efficiency' in feature for feature in self.selected_features):
                self.r.efficiency = self.r.efficiency_(self.r._r)
                self.data[n]['efficiency'] = self.r.efficiency

            if any('Diffusivity' in feature for feature in self.selected_features):
                self.r.velocity = self.r.velocity_(self.r._r, self.r._t)
                self.r.vacf = self.r.stationary_velocity_correlation_(self.r.velocity, self.r._t,np.arange(int(len(self.r.velocity))))
                self.r.diffusivity = self.r.green_kubo_(self.r.velocity, self.r._t,self.r.vacf)
                self.data[n]['diffusivity'] = self.r.diffusivity

            if any('Confinement' in feature for feature in self.selected_features):
                self.r.confinement_probability = self.r.confinement_probability_(self.r._r0, self.r.diffusivity, self.r._t[-1])
                self.data[n]['confinement'] = self.r.confinement_probability
            
        if self.kind=="dir":
            # write the results to a file
            f.write(','.join(self.data[0].keys())+'\n')

            for n in range(0, len(self.data)):
                f.write(','.join(map(str, [*self.data[n].values()]))+'\n')
            
            f.close()

            self.results.delete(0, 'end')
            self.results.insert(0, 'Results saved to {}'.format(f.name))
        else:
            self.results.delete(0, 'end')
            self.results.insert(0, ','.join(map(str, [*self.data[0].values()]))+'\n')

    def compute(self) -> None:
        f = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
        for n, trajectory in enumerate(self.trajectory_list):
            self.r = trajectory
            results = self.r.compute_features()

            f.write(results)

       
        self.results.insert(0, 'Results saved to {}'.format(f.name))

    def get_file(self, kind: str) -> None:
        '''
            get a file if `kind`=="file" or file list from a given directory if `kind`=="dir"
        '''

        files_list = []

        self.kind = kind

        if 'file' in kind:
            path_name = filedialog.askopenfilename(parent=self.app,
                                                      initialdir=self.path,
                                                      title='Please select a file')
        elif 'dir' in kind:
            path_name = filedialog.askdirectory(parent=self.app,
                                                    initialdir=self.path,
                                                    title='Please select a folder')
            # write a file list
            files_list = os.listdir(path_name)

            # filter all csv files
            files_list = [file for file in files_list if '.csv' in file]


        self.path = path_name
        self.files = files_list
        self.entry.delete(0, 100)
        self.entry.insert(0, path_name)

        self.open(kind)

    def save_file(self) -> None:
        f = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
        if f is None:
            return
        f.write(','.join(self.data.keys())+'\n')
        f.write(','.join(self.data.values())+'\n')
        f.close()

    def select_feat(self, button: tk.Widget) -> None:
        if any(button.cget('text') in feature for feature in self.selected_features):
            self.selected_features.remove(button.cget('text'))
            button.deselect()
        else:
            if 'Confinement' in button.cget('text'):
                   # to compute the confinement probability we need another quantities
                   # such as Fractal dimension and diffusivity
                   self.selected_features.append('Fractal dimension')
                   self.selected_features.append('Diffusivity')
                   self.feats_['Fractal dimension'].select()
                   self.feats_['Diffusivity'].select()

            self.selected_features.append(button.cget('text'))
            button.select()

    def select_all(self) -> None:
        self.selected_features = []

        for n, button in enumerate(self.feats_):
            if self.button_select_all.cget('text') == 'Select all':
                self.feats_[button].select()
                self.feats_[button].configure(state='disabled')
                self.selected_features.append(self.feats_[button].cget('text'))

                update_text = 'Deselect all'
            else:
                self.feats_[button].configure(state='active')
                self.feats_[button].deselect()
                update_text = 'Select all'
        self.button_select_all.configure(text=update_text)




if __name__ == '__main__':
    root = ThemedTk(theme="clearlooks")
    tj_Gui = trajpy_gui(root)
    root.mainloop()