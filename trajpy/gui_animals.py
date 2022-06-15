from tkinter import *
from tkinter import filedialog
import animals
import live_tracking
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm

root = Tk()
root.title('TrajPy animals GUI')
root.geometry('470x350')
root.resizable(False, False)




results = []
features = []

def track_function():
    global file_variables
    number_variables = number_entry.get().split(',')
    file_variables = file_entry.get().split(',')
    live_tracking.captura(int(file_variables[0]),file_variables[1],float(number_variables[0]),float(number_variables[1]))

def test_function():
    global file_variables
    import cv2
    file_variables = file_entry.get().split(',')
    cap = cv2.VideoCapture(int(file_variables[0]))
    while(True):
        ret, frame = cap.read()
        if not ret:
            break
        stop_text = 'Press "q" to stop'
        cv2.putText(frame,stop_text,(50,50),cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def open_function():
   root.filename = filedialog.askopenfilename(parent=root,
   initialdir='/'
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
        results.append(animals.displacement(root.filename) + 'cm')
    if 'Center' in features:
        center_variables = center_entry.get().split(',')
        results.append(animals.time_center(float(center_variables[0]),float(center_variables[1]),float(center_variables[2])
        ,float(center_variables[3]),root.filename) + 's')
    if 'Edges' in features:
        edges_variables = edges_entry.get().split(',')
        results.append(animals.time_edges(float(edges_variables[0]),float(edges_variables[1]),float(edges_variables[2])
        ,float(edges_variables[3]),root.filename) + 's')
    label1 = Label(root,text=', '.join(results),font=('Helvetica 16'),background='white')
    label1.place(x=12,y=300)

def clear_function():
    results.clear()
    label1.destroy()
    

def plot_function():
    data = np.loadtxt(root.filename,delimiter=',')
    x = data[:,1]
    y = data[:,2]
    time = data[:,0]/60.0
    plt.figure(dpi=150)
    plt.subplot(121)
    points = np.array([x, y]).T.reshape(-1,1,2)
    segments = np.concatenate([points[:-2],points[1:-1], points[2:]], axis=1)
    lc = LineCollection(segments, cmap=cm.viridis, linewidth=3)
    lc.set_array(time)
    plt.gca().add_collection(lc)
    plt.gca().autoscale()
    cbar = plt.colorbar(lc,orientation="horizontal")
    cbar.set_label(r'$t~$[min]',fontsize=12)
    plt.xlabel(r'$x~$[cm]',fontsize=12)
    plt.ylabel(r'$y~$[cm]',fontsize=12)
    

    plt.subplot(122)
    plt.hist2d(x, y, bins=25,cmap='Blues')
    plt.xlabel(r'$x~$[cm]',fontsize=12)
    plt.ylabel(r'$y~$[cm]',fontsize=12)
    cb = plt.colorbar(orientation="horizontal")
    cb.set_label('Number of occurrences')
    plt.tight_layout()
    plt.show()
    


title_label = Label(root, text="TrajPy", font=("Arial Bold", 35))
title_label.place(x=160,y=0)
version_label = Label(root, text="Animal Tracking", font=("Arial Bold", 10))
version_label.place(x=175,y=50)

track_button = Button(root,text='Track!',command=track_function,font=('Arial',20))
track_button.place(x=10,y=70)

open_button = Button(root,text='Open File',command=open_function,font=('Arial',20))
open_button.place(x=260,y=70)

file_entry = Entry(root,width=28,font=('Arial',12))
file_entry.insert(0,'Insert cam code(int),file name')
file_entry.place(x=10,y=120)

number_entry = Entry(root,width=28,font=('Arial',12))
number_entry.place(x=10,y=150)
number_entry.insert(0,"Insert object's width,height")

center_entry = Entry(root,width=28,font=('Arial',12))
center_entry.place(x=10,y=180)
center_entry.insert(0,'Insert site coordinates (x1,x2,y1,y2)')

edges_entry = Entry(root,width=28,font=('Arial',12))
edges_entry.place(x=10,y=210)
edges_entry.insert(0,"Insert site edges (x1,x2,y1,y2)")

var1 = StringVar()
var2 = StringVar()
var3 = StringVar()

box1 = Checkbutton(root,text='Displacement',variable=var1,onvalue='On',offvalue='Off',command=add_var1,font=('Arial',16))
box1.deselect()
box1.place(x=270,y=120)

box2 = Checkbutton(root,text='Time on site',variable=var2,onvalue='On',offvalue='Off',command=add_var2,font=('Arial',16))
box2.deselect()
box2.place(x=270,y=160)

box3 = Checkbutton(root,text='Time ouside site',variable=var3,onvalue='On',offvalue='Off',command=add_var3,font=('Arial',16))
box3.deselect()
box3.place(x=270,y=200)

compute_button = Button(root,text='Compute',command=compute_function,font=('Arial',20))
compute_button.place(x=10,y=250)

clear_button = Button(root,text='Clear',command=clear_function,font=('Arial',20))
clear_button.place(x=185,y=250)

plot_button = Button(root,text='Plot',command=plot_function,font=('Arial',20))
plot_button.place(x=310,y=250)

test_button = Button(root,text='Cam test',command=test_function,font=('Arial',20))
test_button.place(x=120,y=70)

root.mainloop()
