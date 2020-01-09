import random 
import numpy as np
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def k_center_greedy(points, distance, k):
    centers = set()
    others = set(range(len(points)))

    c_0 = random.randint(0, len(points))
    centers.add(c_0)
    others.remove(c_0)

    logging.info("Finding k centers")

    for _ in tqdm(range(k)):
        distances = dict()
        for point in others:
            min_dist = np.inf
            for center in centers:
                min_dist = min(min_dist, distance(points[point], points[center]))
            distances[point] = min_dist
        new_center = max(distances, key=lambda point: distances[point])
        centers.add(new_center)
        others.remove(new_center)
    
    return centers
    
