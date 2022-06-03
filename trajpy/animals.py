import numpy as np



def displacement(file):
    file = np.loadtxt(str(file),delimiter=',')
    dis = np.sum(np.sqrt(np.sum(np.diff(file[:,1:],axis=0)**2,axis=1)))
    return str(np.round(dis,2))

def time_center(x1,x2,y1,y2,file):
    tc = 0
    file = np.loadtxt(str(file),delimiter=',')

    for i in range (file.shape[0]-1):
        if (file[i,1] > x1 and file[i,1] < x2 and file[i,2] > y1 and file[i,2] < y2) and (file[i+1,1] > x1 and file[i+1,1] < x2 
                                                                                    and file[i+1,2] > y1 and file[i+1,2] < y2):
            tc += file[i+1,0] - file[i,0]
    return str(np.round(tc,2))

def time_edges(x1,x2,y1,y2,file):
    file = np.loadtxt(str(file),delimiter=',')

    te = 0
    for i in range(file.shape[0]-1):
        if (file[i,1] < x1 or file[i,1] > x2): 
            if (file[i+1,1] < x1 or file[i+1,1] > x2): 
                te += file[i+1,0] - file[i,0]
            elif (file[i+1,1] > x1 and file[i+1,1] < x2) and (file[i+1,2] < y1 or file[i+1,2] > y2): 
                te += file[i+1,0] - file[i,0]

        elif (file[i,1] > x1 and file[i,1] < x2) and (file[i,2] < y1 or file[i,2] > y2): 
            if (file[i+1,1] < x1 or file[i+1,1] > x2): 
                te += file[i+1,0] - file[i,0]
            elif (file[i+1,1] > x1 and file[i+1,1] < x2) and (file[i+1,2] < y1 or file[i+1,2] > y2): 
                te += file[i+1,0] - file[i,0]
    return str(np.round(te,2))



