import numpy as np
from bs4 import BeautifulSoup, Comment
import os
import cv2

output_path = './output/flame_2d/19_3'
input_path = './input/3_cameras/19/'

def string2float(string):
    strs = string.split('e')
    if len(strs) == 1:
        return float(strs[0])
    return float(strs[0]) * 10**float(strs[1])

def trans_matrix(string):
    mtx  = np.zeros((4, 4))
    strs = string.split(' ')
    for i in range(len(strs)):
        mtx[int(i/4), i%4] = string2float(strs[i])
    return mtx

def Rt2translation(Rt):
	return Rt[:3, 3]

def dist2orgin(t):
	return (t[0]**2 + t[1]**2 + t[2]**2)**0.5

def dist(t1, t2):
	return ((t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 + (t1[2]-t2[2])**2)**0.5

def readOBJpc(path):
    file  = open(path, 'r')
    lines = file.readlines()
    pts   = np.ones((len(lines), 4))
    for i in range(len(lines)):
        nums = lines[i].split('\n')[0].split(' ')
        #print(nums)
        pts[i, :3] = [float(nums[1]), float(nums[2]), float(nums[3])]
    return pts


folder = os.path.join(input_path,'xml')
save_folder = os.path.join(output_path, 'pi')

for name in os.listdir(folder):
    person = name.split('.')[0]

    xml_name = os.path.join(folder, name)
    with open(xml_name, 'r') as f:
        data = f.read()
    Bs_data = BeautifulSoup(data, "xml")
    sensors = Bs_data.find_all('sensor')
    cameras = Bs_data.find_all('camera')

    f  = float(Bs_data.find_all('f')[0].text)
    cx = float(Bs_data.find_all('cx')[0].text) + 2896/2
    cy = float(Bs_data.find_all('cy')[0].text) + 4344/2

    #center = np.array([0.0827, -0.2641, -4.2457])

    #imgs = ['17.jpg', '21.jpg']
    #imgs = ['40.jpg', '44.jpg', '45.jpg']
    imgs = ['1.jpg', '8.jpg', '10.jpg']

    Rt = []
    for i in range(len(imgs)):
        Rt.append(trans_matrix(cameras[i].find('transform').text))

    K = np.zeros((4, 4))
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = cx
    K[1, 2] = cy
    K[2, 2] = 1
    K[3, 3] = 1

    print('--------')
    print(K)
    print('---------')

    #pw = readOBJpc('./flame_3d/4044_fit_scan_result.obj')
    pw = readOBJpc('./input/3_cameras/19/proj_flame.obj')
    #print(pw.shape)
    #pw = readOBJpc('./scans/4044_agisoft_raw.obj')
    # pw = readOBJpc('flame1797.obj')
    # pw = readOBJpc('DADcali404445.obj')

    np.save(os.path.join(output_path,'pw_flame.npy'), pw)

    #print(K)
    #print(Rt)
    #print(pw)


    for j in range(len(imgs)):
        print('****************', imgs[j])
        print(Rt[j])
        pc = np.matmul(np.linalg.inv(Rt[j]), pw.T)
        pi = np.matmul(K, pc)

        # xi = pi[0] / pi[3]
        # yi = pi[1] / pi[3]
        # zi = pi[2] / pi[3]
        # xi = xi / zi
        # yi = yi / zi

        pi = pi / pi[3, :]
        pi = pi[:2, :] / pi[2, :]
        pi = pi.T
        np.save(os.path.join(output_path, imgs[j].split('.')[0]+'.npy'), pi)

        print(np.min(pi[:, 0]), np.max(pi[:, 0]), np.mean(pi[:, 0]))
        print(np.min(pi[:, 1]), np.max(pi[:, 1]), np.mean(pi[:, 1]))

        np.save(os.path.join(save_folder, imgs[j].split('.')[0]+'.npy'), pi)


        img = cv2.imread(os.path.join(input_path, imgs[j]))

        for i in range(pi.shape[0]):
            try:
                x = int(pi[i, 0])
                y = int(pi[i, 1])
                img[y-10:y+10, x-10:x+10, :] = [0, 255, 0]
            except:
                pass

        cv2.imwrite(os.path.join(output_path, 'visu_'+imgs[j]), img)