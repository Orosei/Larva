import argparse
import numpy as np


parser = argparse.ArgumentParser(description="Parameters of the project")
parser.add_argument("--dataset_name", type=str, default="CFD")
parser.add_argument("--alpha", type=np.float32, default=1.0, help="the weight of internal 1st")
parser.add_argument("--beta", type=np.float32, default=1.0, help="the weight of internal 2nd")
parser.add_argument("--gamma", type=np.float32, default=0.125, help="the weight of external energy1")
parser.add_argument("--zeta", type=np.float32, default=0.1, help="the weight of external energy2")
parser.add_argument("--delta", type=np.float32, default=15.0, help="the weight of repulsive")
parser.add_argument("--eta", type=np.float32, default=20.0, help="the weight of gravity")
parser.add_argument("--radius", type=np.int32, default=10, help="the radius search nearby")

# parser.add_argument("--alpha", type=np.float32, default=1.0, help="the weight of internal 1st")
# parser.add_argument("--beta", type=np.float32, default=1.0, help="the weight of internal 2nd")
# parser.add_argument("--gamma", type=np.float32, default=50.0, help="the weight of external energy1")
# parser.add_argument("--zeta", type=np.float32, default=0.5, help="the weight of external energy2")
# parser.add_argument("--delta", type=np.float32, default=0.0, help="the weight of repulsive")
# parser.add_argument("--eta", type=np.float32, default=20.0, help="the weight of gravity")
# parser.add_argument("--radius", type=np.int32, default=10, help="the radius search nearby")

args = parser.parse_args()
