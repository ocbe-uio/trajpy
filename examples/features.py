import trajpy.trajpy as tj

filename = '../data/samples/sample.csv'
r = tj.Trajectory(filename, skip_header=1, delimiter=',')

r.compute_features()

print(r.efficiency,
      r.asymmetry,
      r.gaussianity,
      r.straightness,
      r.gyration_radius,
      )