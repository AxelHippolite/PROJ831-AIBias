import json
import multiprocessing
import pandas as pd
from tqdm import tqdm
import glob
import cv2
import numpy as np
import porespy.metrics as pm
from scipy.stats import linregress
    
def process_json_file(file_path):
    with open(file_path, 'r') as f:
        json_data = json.load(f)
    labels = [obj['label']  for obj in json_data['objects']]
    count_res = {label : labels.count(label) for label in set(labels)}
    city = file_path.split('\\')[1]
    count_res['city'] = city
    poly_lists = {city : {}}
    for obj in json_data['objects']:
        label = obj['label']
        if label in poly_lists[city]:
            poly_lists[city][label].append(obj['polygon'])
        else:
            poly_lists[city][label] = [obj['polygon']]
    return pd.DataFrame.from_dict([count_res]), poly_lists

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
    
    file_list = file_list = glob.glob('data/gtFine/train/*/*.json') + glob.glob('data/gtFine/val/*/*.json')

    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)
    results = pool.map(process_json_file, file_list)
    pool.close()
    pool.join()

    df_count = results[0][0]
    merged_dict = {}
    
    for res in tqdm(results[1:]):
        df_count = pd.concat([df_count, res[0]], sort=False)
        city = list(res[1].keys())[0]
        if city in merged_dict:
            for key, value in res[1][city].items():
                    if key in merged_dict[city]:
                        merged_dict[city][key].extend(value)
                    else:
                        merged_dict[city][key] = value
        else:
            merged_dict[city] = res[1][city]
    
    city = df_count.pop('city')
    df_count.insert(0, 'city', city)
    df_count.to_csv('output/labels_count.csv', index=False)
    print('Dataframe saved to : output/labels_count.csv')
    
    with open("output/polygons.json", "w") as outfile:
        json.dump(merged_dict, outfile)
    print('Json file saved to : output/polygons.json')
    
    file_list = glob.glob('data/gtFine/train/*/*color*') + glob.glob('data/gtFine/val/*/*color*')
    res = {}
    for file in tqdm(file_list):
        city, fd = fractal_dimension(file)
        if city in res:
            res[city].append(fd)
        else:
            res[city] = [fd]
    df_fractal = pd.DataFrame(res)
    df_fractal.to_csv('output/fractal_dimension.csv', index=False)