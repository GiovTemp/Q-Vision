import numpy as np

m = np.random.uniform((4, 4))

print(np.sqrt(np.sum(np.sum(np.multiply(m, m)))))

print(np.linalg.norm(m))
