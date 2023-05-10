import multiprocessing
import pandas as pd
from tqdm import tqdm
import glob
import cv2
import numpy as np
import porespy.metrics as pm
from scipy.stats import linregress

def fractal_dimension(path):
    city = path.split('\\')[1]
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    edges = cv2.Canny(img, 100, 200)
    box_count = pm.boxcount(edges)
    x = np.log(box_count.size)
    y = np.log(box_count.count)
    slope = linregress(x, y).slope
    fd = -slope
    return city, fd

if __name__ == '__main__':
    file_list = glob.glob('data/gtFine/train/*/*color*') + glob.glob('data/gtFine/val/*/*color*')

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)
    results = pool.map(fractal_dimension, file_list)
    pool.close()
    pool.join()
        
    fractals = {}
    for res in tqdm(results):
        if res[0] in fractals:
            fractals[res[0]].append(res[1])
        else:
            fractals[res[0]] = [res[1]]
    df_fractal = pd.DataFrame.from_dict(fractals, orient='index').T
    df_fractal.to_csv('output/fractal_dimension.csv', index=False)