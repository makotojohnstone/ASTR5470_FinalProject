'''
Functions used final project
'''
# import neccesary packages here
import numpy as np
import matplotlib.pyplot as plt
import astropy
from photutils.psf import CircularGaussianPRF
import matplotlib.colors as colors
import cv2
from itertools import combinations
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import math

class SNe_pop:
    """ Simulating a population of supernovae and supernova remnants """

    def __init__(self, rate):
        """ Initialize the dice class """
        self.n = int(rate*1000*0.025) #rounding down
        self.lumin = None
        self.diam = None
        self.x = None
        self.y = None

    def luminosities(self, lumin_range, beta = -2.19):
        '''
        '''
        PDF = lumin_range**beta/self.n
        PDF = PDF/np.sum(PDF) #normalize the probability distribution
        self.lumin = np.random.choice(lumin_range, size=self.n, replace=True, p=PDF)

    def diameter(self, A=40, alpha=-4/9):
        self.diam = A * self.lumin**alpha + np.random.normal(0, 0.2, self.n)

    def position(self, N, scale_height=None, dist = 'exp'):
        '''
        Note that the scale height must be in units of pixels

        Sources can overlap but not have the same position. Note that this requires a fairly large N. 
        We recommend at least N > 10 times the number of SNe
        '''
        if dist == 'exp':
            if scale_height == None:
                print('Must input scale_height parameter')
            else:
                
                # Randomly select a radius assuming an exponential disk profile
                r = np.random.exponential(scale=scale_height, size=self.n)
                
                # Randomly select an angle
                theta = np.random.uniform(0, 2 * np.pi, self.n)

                # Determine position in cartesian coordinates
                x = N/2 - r * np.cos(theta)
                y = N/2 - r * np.sin(theta)
                    
                self.x, self.y = x, y

        elif dist == 'gauss':
            x, y = np.random.normal(loc=N/2, scale=N/6, size=(2, self.n))
            pos = np.array([[x[i], y[i]] for i in range(self.n)])

            for i in range(self.n):
                while np.any(pos[i] < 0) or np.any(pos[i] > N): #make sure the position is in the FOV
                    pos = np.delete(pos, i, axis=0)
                    pos = np.append(pos, np.random.normal(loc=N/2, scale=N/6, size=(1, 2)), axis=0)  

            self.x, self.y = pos[:,0], pos[:,1]
            
        elif dist == 'uniform':
            pos = np.random.uniform(0, N, size=(self.n, 2))
            self.x, self.y = pos[:,0], pos[:,1]

        else:
            print('Must select between exponential disk (exp), gaussian (gauss), or uniform distribution (uniform)')
        
    def make_data(self, i, N, px):
        yy, xx = np.mgrid[:N, :N]
        data = np.zeros((N,N))
        data += CircularGaussianPRF(flux=self.lumin[i], x_0=self.x[i], y_0=self.y[i], fwhm=1.177*self.diam[i]/px)(xx,yy)
        return data

def aperture_pos(width_px, height_px, radius_px, step_px=15):
    """
    Finds all valid integer center positions for a circle 
    that fits completely within a grid.

    All inputs are in pixels, must be integeres
    """
    positions = []
    
    # The center must be at least 'radius' away from the edge
    for x in range(radius_px, width_px - radius_px + 1, step_px):
        for y in range(radius_px, height_px - radius_px + 1, step_px):
            positions.append((x, y))
            
    return positions

def check_overlap(centers, radius_px):
    n = len(centers)

    # Normalize radii
    if isinstance(radius_px, (int, float)):
        radii = [radius_px] * n
    else:
        radii = radius_px

    # Build adjacency list (direct overlaps only)
    graph = {i: set() for i in range(n)}

    def overlaps(i, j):
        x1, y1 = centers[i]
        x2, y2 = centers[j]
        r1, r2 = radii[i], radii[j]

        dx = x2 - x1
        dy = y2 - y1

        return dx * dx + dy * dy <= (r1 + r2) ** 2

    # Create edges only for direct overlaps
    for i in range(n):
        for j in range(i + 1, n):
            if overlaps(i, j):
                graph[i].add(j)
                graph[j].add(i)

    # DFS to count components (isolated nodes included)
    visited = set()
    groups = 0

    for i in range(n):
        if i not in visited:
            groups += 1
            stack = [i]

            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                for nei in graph[node]:
                    if nei not in visited:
                        stack.append(nei)

    return groups

def measure_lumin(aperture, N, radius_px, data, criteria):
    mask = np.zeros((N, N), dtype=np.uint8)
    circle = cv2.circle(mask, aperture, int(radius_px), 1, -1)
    aperture_lumin = data*circle

    if sum(sum(aperture_lumin)) > criteria:
        bright_aperture = aperture
    else:
        bright_aperture = (np.nan, np.nan)
    
    return sum(sum(aperture_lumin)), bright_aperture


def simulate_sne(rate, radius_pc, lumin_range, criteria, FOV = 200, N=500, dist='exp', scale_height_pc = 100, step_pc=10, A=40, alpha=-4/9, beta = -2.19, visualize=False):
    '''
    Aperture radius in parsecs
    FOV in parsecs
    step in parsecs
    scale_height in parsecs
    N in number of pixels, must be an integer
    '''
    ### Convert values into pixel units
    px = FOV/N #pixel scale
    radius_px = radius_pc/px
    step_px = step_pc/px
    scale_height_px = scale_height_pc/px

    ### Simulate the population
    pop = SNe_pop(rate)
    pop.luminosities(lumin_range, beta = beta)
    pop.diameter(A=A, alpha = alpha)
    pop.position(N, scale_height_px, dist = 'exp')

    ### Make the data
    data = sum((pop.make_data)(x, N, px) for x in range(pop.n))

    if visualize: ### A quick visual
        fig = plt.figure(figsize=(8,8))
        plt.imshow(data, norm=colors.LogNorm(vmin=min(pop.lumin)/1000, vmax=max(pop.lumin), clip=True), cmap='hot', extent=[-N/20, N/20, -N/20, N/20])
        plt.colorbar(label='Luminosity')
        plt.title('Simulated SNe/SNR population')
        fig.savefig('./Simulated_pop.png', bbox_inches='tight')

    ### Add apertures and measure luminosities for each
    apertures = aperture_pos(N,N, int(radius_px), step_px=int(step_px))
    measured = [measure_lumin(aperture, N, radius_px, data, criteria) for aperture in apertures]
    total_lumin = np.array([sublist[0] for sublist in measured])
    bright_apertures = np.array([sublist[1] for sublist in measured])[~np.isnan(np.array([sublist[1] for sublist in measured])).any(axis=1)]

    ### Determine how many (if any) of the apertures were overlapping, subtract those from the total count
    count = check_overlap(bright_apertures, radius_px) 

    return total_lumin, count, len(apertures), pop


def run_mc(iterations, n_jobs, rate, radius_pc, lumin_range, criteria, FOV = 200, N=2000, dist='exp', scale_height_pc = 100, step_pc=10, visualize=False):
    ''
    ''
    results = Parallel(n_jobs=n_jobs)(delayed(simulate_sne)(rate=rate, radius_pc = radius_pc, lumin_range=lumin_range, criteria=criteria, FOV=FOV, N=N, dist='exp', scale_height_pc=scale_height_pc, step_pc=step_pc, visualize=False) for x in range(iterations))
    
    total_luminosities = [item[0] for item in results]
    count = [item[1] for item in results]
    apertures = [item[2] for item in results]
    population = [item[3] for item in results]

    total_luminosities_cat = np.concatenate(total_luminosities)
    return total_luminosities_cat, count, apertures, population

        