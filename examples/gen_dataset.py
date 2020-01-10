import trajpy.trajpy as tj
import trajpy.traj_generator as tjg

trajectory = 'anomalous'
r = np.zeros((250, 4))
n_steps = 250
n_samples = 5000
dt = 1.0
D = 100.

with open(trajectory+'.csv', 'w+', buffering=1) as f:
    f.write('label,alpha,ratio,df,anisotropy,kurtosis,straightness,gaussianity,trappedness,diffusivity,efficiency\n')
    
    if 'normal' in trajectory:
        diffusivity = np.array([10., 100., 1000., 10000.,])
        for value in diffusivity:
            x1, y1 = tjg.normal_diffusion(n_steps, n_samples, 1.0, 0., value, dt)
            x2, y2 = tjg.normal_diffusion(n_steps, n_samples, 1.0, 0., value, dt)
            x3, y3 = tjg.normal_diffusion(n_steps, n_samples, 1.0, 0., value, dt)
            for n in range(0, n_samples):
                r[:, 0] = x1[:]
                r[:, 1] = y1[:, n]
                r[:, 2] = y2[:, n]
                r[:,  3] = y3[:, n]
                traj = tj.Trajectory(r)
                try:
                    features = traj.compute_features()
                    f.write(trajectory+','+features+'\n')
                    f.flush()
                except:
                    print('error computing features.')
    
    elif 'confined' in trajectory:
        radius = np.array([5., 10., 20.])
        for value in radius:
    
            x1, y1 = confined_diffusion(value, n_steps, n_samples, 1.0, 0.0, D, dt)
            x2, y2 = confined_diffusion(value, n_steps, n_samples, 1.0, 0.0, D, dt)
            x3, y3 = confined_diffusion(value, n_steps, n_samples, 1.0, 0.0, D, dt)
            for n in range(0, n_samples):
                r[:, 0] = x1[:]
                r[:, 1] = y1[:, n]
                r[:, 2] = y2[:, n]
                r[:, 3] = y3[:, n]
                traj = tj.Trajectory(r)
                try:
                    features = traj.compute_features()
                    f.write(trajectory+','+features+'\n')
                    f.flush()
                except:
                    print('error computing features.')
    
    elif 'superdiffusion' in trajectory:
        n_steps = 250
        n_samples = 5000
        dt = 1.0
        velocity = np.array([0.1, 0.2, 0.3, 0.4])
        D = 100.
        ballistic_traj = np.zeros((n_steps, 4))
    
        for i in range(0, len(velocity)):
            ballistic_traj[:, i] = tjg.superdiffusion(velocity[i], n_steps, 0., dt)[1]
            
            for i in range(0, len(velocity)):
    
                x1, y1 =  tjg.normal_diffusion(n_steps, n_samples, 1.0, 0., 100., dt) + (0., ballistic_traj[:, i])
                x2, y2 =  tjg.normal_diffusion(n_steps, n_samples, 1.0, 0., 100., dt)
                x3, y3 =  tjg.normal_diffusion(n_steps, n_samples, 1.0, 0., 100., dt)
    
                for n in range(0, n_samples):
                    r[:, 0] = x1[:]
                    r[:, 1] = y1[:, n] 
                    r[:, 2] = y2[:, n] 
                    r[:, 3] = y3[:, n] 
                    traj = tj.Trajectory(r)
                    try:
                        features = traj.compute_features()
                        f.write(trajectory+','+features+'\n')
                        f.flush()
                    except:
                        print('error computing features.')
        
    if 'anomalous' in trajectory:
        alpha = np.array([1.4, 1.8, 2.0])
    
        for value in alpha:
            x1, y1 = tjg.anomalous_diffusion(n_steps, n_samples, dt, value)
            x2, y2 = tjg.anomalous_diffusion(n_steps, n_samples, dt, value)
            x3, y3 = tjg.anomalous_diffusion(n_steps, n_samples, dt, value)
            for n in range(0, n_samples):
                r[:, 0] = x1[:]
                r[:, 1] = y1[:, n]
                r[:, 2] = y2[:, n]
                r[:,  3] = y3[:, n]
                traj = tj.Trajectory(r)
                try:
                    features = traj.compute_features()
                    f.write(trajectory+','+features+'\n')
                    f.flush()
                except:
                    print('error computing features.')
    
