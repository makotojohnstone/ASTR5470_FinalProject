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
import sampling as samp
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

    def diameter(self, A=1.15e12, alpha=-4/9):
        self.diam = A * self.lumin**alpha + np.random.normal(0, 0.2, self.n)

        mask = (self.diam < 0.2) | (self.diam > 2)
        while np.any(mask):
            n_bad = np.sum(mask)
            new_vals = A * self.lumin[mask]**alpha + np.random.normal(0, 0.1, n_bad)
            self.diam[mask] = new_vals
            mask = (self.diam < 0.2) | (self.diam > 2)

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

                pos = np.array([[x[i], y[i]] for i in range(self.n)])
                unique_pos = np.unique(pos, axis = 0)

                ## Make sure all values are unique (highly improbable that the same position is pulled twice)
                while len(unique_pos) != self.n:
                    r = np.random.exponential(scale=scale_height, size=1)
                    theta = np.random.uniform(0, 2 * np.pi, 1)
                    
                    np.append(x, N/2 - r * np.cos(theta))
                    np.append(y, N/2 - r * np.sin(theta))

                    pos = np.array([[x[i], y[i]] for i in range(self.n)])
                    unique_pos = np.unique(pos, axis = 0)
                    
                self.x, self.y = unique_pos[:,0], unique_pos[:,1]

        elif dist == 'gauss':
            x, y = np.random.normal(loc=N/2, scale=N/6, size=(2, self.n))
            pos = np.array([[x[i], y[i]] for i in range(self.n)])

            for i in range(self.n):
                while np.any(pos[i] < 0) or np.any(pos[i] > N): #make sure the position is in the FOV
                    pos = np.delete(pos, i, axis=0)
                    pos = np.append(pos, np.random.normal(loc=N/2, scale=N/6, size=(2, 1)))   

            unique_pos = np.unique(pos, axis = 0) #check if every value is unique
            while len(unique_pos) != self.n: 
                new_value = np.array([-1, -1])
                while np.any(new_value < 0) or np.any(new_value > N): #confirm that it's in the field of view
                    new_value = np.random.normal(loc=N/2, scale=N/6, size=(2, 1))
                
                pos = np.append(pos, new_value) #add a new one
                unique_pos = np.unique(pos, axis = 0)
                
        
            for i in range(self.n):
                while pos[i] in np.delete(pos, i, axis=0): #if the element is repeated
                    pos = np.delete(pos, i, axis=0) #delete the repeating element
                    pos = np.append(pos, np.random.normal(loc=N/2, scale=N/6, size=(2, 1))) #add a new one
                                    
                while np.any(pos[i] < 0) or np.any(pos[i] > N): #make sure the position is in the FOV
                    pos = np.delete(pos, i, axis=0)
                    pos = np.append(pos, np.random.normal(loc=N/2, scale=N/6, size=(2, 1)))    

            self.x, self.y = pos[:,0], pos[:,1]
            
        elif dist == 'uniform':
            pos = np.random.uniform(0, N, size=(self.n, 2))
            unique_pos = np.unique(pos, axis = 0) #check if every value is unique
            
            while len(unique_pos) != self.n: 
                pos = np.append(pos, np.random.uniform(0, N, size=(1, 2))) #add a new one
                unique_pos = np.unique(pos, axis = 0)

            self.x, self.y = unique_pos[:,0], unique_pos[:,1]

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
    if len(centers) == 0:
        return 0

    tree = cKDTree(centers)
    pairs = tree.query_pairs(r=2*radius_px)

    overlapped = set(i for pair in pairs for i in pair)
    return len(centers) - len(overlapped)

def measure_lumin(aperture, N, radius_px, data, criteria):
    mask = np.zeros((N, N), dtype=np.uint8)
    circle = cv2.circle(mask, aperture, int(radius_px), 1, -1)
    aperture_lumin = data*circle

    if sum(sum(aperture_lumin)) > criteria:
        bright_aperture = aperture
    else:
        bright_aperture = (np.nan, np.nan)
    
    return sum(sum(aperture_lumin)), bright_aperture


def simulate_sne(rate, radius_pc, lumin_range, criteria, FOV = 200, N=2000, dist='exp', scale_height_pc = 100, step_pc=1.5, visualize=False):
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
    pop.luminosities(lumin_range)
    pop.diameter()
    pop.position(N, scale_height_px, dist = 'exp')

    ### Make the data
    data = sum((pop.make_data)(x, N, px) for x in range(pop.n))

    if visualize: ### A quick visual
        plt.imshow(data, norm=colors.LogNorm(vmin=min(pop.lumin)/1000, vmax=max(pop.lumin), clip=True), cmap='hot', extent=[-N/20, N/20, -N/20, N/20])
        plt.colorbar(label='Luminosity')
        plt.title('Simulated SNe/SNR population')

    ### Add apertures and measure luminosities for each
    apertures = aperture_pos(N,N, int(radius_px), step_px=int(step_px))
    measured = [measure_lumin(aperture, N, radius_px, data, criteria) for aperture in apertures]
    total_lumin = np.array([sublist[0] for sublist in measured])
    bright_apertures = np.array([sublist[1] for sublist in measured])[~np.isnan(np.array([sublist[1] for sublist in measured])).any(axis=1)]

    ### Determine how many (if any) of the apertures were overlapping, subtract those from the total count
    count = check_overlap(bright_apertures, radius_px) 

    return total_lumin, count, pop


def run_mc(iterations, n_jobs, rate, radius_pc, lumin_range, criteria, FOV = 200, N=2000, dist='exp', scale_height_pc = 100, step_pc=1.5, visualize=False):
    ''
    ''
    results = Parallel(n_jobs=n_jobs)(delayed(simulate_sne)(rate=2, radius_pc = 10, lumin_range=lumin_range, criteria=1e4, FOV=200, N=500,
                                                        dist='exp', scale_height_pc=100, step_pc=5, visualize=False) for x in range(iterations))
    total_luminosities = [item[0] for item in results]
    count = [item[1] for item in results]
    population = [item[2] for item in results]

    total_luminosities_cat = np.concatenate(total_luminosities)
    return total_luminosities_cat, count, population

        