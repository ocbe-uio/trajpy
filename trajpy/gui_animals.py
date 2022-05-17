from tkinter import *
from tkinter import filedialog
import animals
import live_tracking
import numpy as np
import matplotlib.pyplot as plt

root = Tk()
root.title('TrajPy animals GUI')
root.geometry('400x400')
root.resizable(False, False)

results = []
features = []

def track_function():
    live_tracking.captura(file_entry.get(),number_entry.get())

def open_function():
   root.filename = filedialog.askopenfilename(parent=root,
   initialdir='/home/secundario/Documentos/mestrado/projeto/code testing/data'
   ,title='Please select a file')

def add_var1():
    if var1.get() == 'On':
        features.append('Displacement')
    elif var1.get() == 'Off':
        features.remove('Displacement')

def add_var2():
    if var2.get() == 'On':
        features.append('Center')
    elif var2.get() == 'Off':
        features.remove('Center')

def add_var3():
    if var3.get() == 'On':
        features.append('Edges')
    elif var3.get() == 'Off':
        features.remove('Edges')

def compute_function():
    global label1
    if "Displacement" in features:
        results.append(animals.displacement(root.filename))
    if 'Center' in features:
        center_variables = center_entry.get().split(',')
        results.append(animals.time_center(float(center_variables[0]),float(center_variables[1]),float(center_variables[2])
        ,float(center_variables[3]),root.filename))
    if 'Edges' in features:
        edges_variables = edges_entry.get().split(',')
        results.append(animals.time_edges(float(edges_variables[0]),float(edges_variables[1]),float(edges_variables[2])
        ,float(edges_variables[3]),root.filename))
    label1 = Label(root,text=','.join(results),font=('Helvetica 12 bold'),background='white')
    label1.place(x=10,y=240)

def clear_function():
    results.clear()
    label1.destroy()
    

def plot_function():
    data = np.loadtxt(root.filename,delimiter=',')
    x = data[:,1]
    y = data[:,2]
    r = np.sqrt(np.sum(data[:,1:]**2,axis=1))
    time = data[:,0]
    plt.figure(figsize=(8,5),dpi=150)
    plt.subplot(121)
    cm = plt.cm.get_cmap('viridis')
    sc = plt.scatter(x,y,c=time,vmin=min(time), vmax=max(time+5), s=35, cmap=cm)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$t~$[s]',fontsize=12)
    plt.xlabel(r'$x~$[cm]',fontsize=12)
    plt.ylabel(r'$y~$[cm]',fontsize=12)
    plt.subplot(122)
    plt.hist2d(x, y, bins=50,cmap='plasma')
    plt.xlabel(r'$x~$[cm]',fontsize=12)
    plt.ylabel(r'$y~$[cm]',fontsize=12)
    cb = plt.colorbar()
    cb.set_label('# of occurrences')
    plt.tight_layout()
    plt.show()



title_label = Label(root, text="TrajPy", font=("Arial Bold", 28))
title_label.place(x=160,y=10)
version_label = Label(root, text="Animal Tracking", font=("Arial Bold", 10))
version_label.place(x=165,y=50)

track_button = Button(root,text='Track!',command=track_function)
track_button.place(x=30,y=70)

open_button = Button(root,text='Open File',command=open_function)
open_button.place(x=100,y=70)

file_entry = Entry(root,width=30)
file_entry.insert(0,'Insert file name')
file_entry.place(x=10,y=100)

number_entry = Entry(root,width=30)
number_entry.place(x=10,y=120)
number_entry.insert(0,'Insert object size')

center_entry = Entry(root,width=30)
center_entry.place(x=10,y=140)
center_entry.insert(0,'Insert center coordinates')


edges_entry = Entry(root,width=30)
edges_entry.place(x=10,y=160)
edges_entry.insert(0,'Insert edges coordinates')

var1 = StringVar()
var2 = StringVar()
var3 = StringVar()

box1 = Checkbutton(root,text='Displacement',variable=var1,onvalue='On',offvalue='Off',command=add_var1)
box1.deselect()
box1.place(x=200,y=100)

box2 = Checkbutton(root,text='Time at Center',variable=var2,onvalue='On',offvalue='Off',command=add_var2)
box2.deselect()
box2.place(x=200,y=120)

box3 = Checkbutton(root,text='Time at Edges',variable=var3,onvalue='On',offvalue='Off',command=add_var3)
box3.deselect()
box3.place(x=200,y=140)

compute_button = Button(root,text='Compute',command=compute_function)
compute_button.place(x=10,y=210)

clear_button = Button(root,text='Clear',command=clear_function)
clear_button.place(x=90,y=210)

plot_button = Button(root,text='Plot',command=plot_function)
plot_button.place(x=150,y=210)

root.mainloop()