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
import csv
from joblib import Parallel, delayed

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
        Samples luminosities based on a luminosity function of the form n(L) = AL^beta

        lumin_range: luminosity range of simulated population in units of 10^24 erg/s/Hz
        '''
        PDF = lumin_range**beta #the luminosity function
        PDF = PDF/np.sum(PDF) #normalize the probability distribution
        self.lumin = np.random.choice(lumin_range, size=self.n, replace=True, p=PDF) #randomly sample from the function

    def diameter(self, A=40, alpha=-4/9):
        '''
        Converts luminosities into diameters given a user-inputted power-law relation (A*L^alpha) with random Gaussian noise added

        A (int or float): Scaling factor for luminosity-diameter relation
        alpha (int or float): power law index for luminosity-diameter relation
        '''
        self.diam = A * self.lumin**alpha + np.random.normal(0, 0.1, self.n)

    def position(self, N, scale_height=None, dist = 'exp'):
        '''
        Randomly samples source positions based on user-defined distribution. Options include dist = 'exp' (exponential disk', 'gauss'
        (gaussian), and 'uniform' (uniform).  

        Inputs:

        N (int) : the number of pixels N for a modeled spatial region of size NxN. 
        scale_height (int): the scale_height in pixels, required for an exponential disk distribution. 
        dist (String): Defines the spatial distribution of the sources. Options are 'exp' (exponential disk), 'gauss' (gaussian), and 'uniform', (uniform).  
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
            # randomly select a position assuming a gaussian distribution
            x, y = np.random.normal(loc=N/2, scale=N/6, size=(2, self.n))
            pos = np.array([[x[i], y[i]] for i in range(self.n)])

            #make sure the position is in the FOV
            for i in range(self.n):
                while np.any(pos[i] < 0) or np.any(pos[i] > N): # if outside of the field of view
                    pos = np.delete(pos, i, axis=0)
                    pos = np.append(pos, np.random.normal(loc=N/2, scale=N/6, size=(1, 2)), axis=0)  

            self.x, self.y = pos[:,0], pos[:,1]
            
        elif dist == 'uniform':
            # Randomly select a position assuming a uniform distribution
            pos = np.random.uniform(0, N, size=(self.n, 2))
            self.x, self.y = pos[:,0], pos[:,1]

        else:
            print('Must select between exponential disk (exp), gaussian (gauss), or uniform distribution (uniform)')
        
    def make_data(self, i, N, px): 
        '''
        Make the 2-dimensional simulated data given the luminosities, diameters, and source positions of the SNe/SNR population

        Inputs;
        i (int): the index of the source
        N (int): the number of pixels per dimension of the modeled region
        px (int or float): pixel scale of the model (parsecs per pixel)

        Returns the simulated data
        '''
        yy, xx = np.mgrid[:N, :N] 
        data = np.zeros((N,N)) #make an empty grid
        data += CircularGaussianPRF(flux=self.lumin[i], x_0=self.x[i], y_0=self.y[i], fwhm=1.177*self.diam[i]/px)(xx,yy) #add source
        return data

def aperture_pos(width_px, height_px, radius_px, step_px):
    """
    Finds all valid integer center positions for a circle 
    that fits completely within a grid.

    Inputs:
    width_px (int): width of the modeled region in pixels
    height_px (int): height of the modeled region in pixels
    radius_px (int or float): radius of the aperture in pixels
    step_px (int): distance between aperture centers in piels

    All inputs are in pixels, must be integers

    Returns a list of positions to place the apertures
    """
    positions = []
    
    # The center must be at least 'radius' away from the edge
    for x in range(radius_px, width_px - radius_px + 1, max(1,int(step_px))):
        for y in range(radius_px, height_px - radius_px + 1, max(1,int(step_px))):
            positions.append((x, y))
            
    return positions

def check_overlap(centers, radius_px):
    """
    Count how many independent groups of overlapping circular apertures exist. 
    This is to ensure none of the measurements above the user-inputted threshold are redundant.

    Inputs:
    centers (list of tuples): (x, y) pixel coordinates for each aperture center.

    radius_px (int, float): aperture radius in pixels.
        
    Returns the number of "groups"  (# of groups of overlapping apertures + # of isolated apertures)
    """

    n = len(centers)

    # Keep track of which apertures directly overlap
    overlap_map = {i: set() for i in range(n)}

    # Compare every pair of apertures
    for i in range(n):
        x1, y1 = centers[i]
        r1 = radius_px

        for j in range(i + 1, n):
            x2, y2 = centers[j]
            r2 = radius_px

            # Squared distance between centers
            dx = x2 - x1
            dy = y2 - y1
            distance_squared = dx**2 + dy**2

            # Squared overlap threshold
            overlap_distance_squared = (r1 + r2) ** 2

            # Record overlap connection
            if distance_squared <= overlap_distance_squared:
                overlap_map[i].add(j)
                overlap_map[j].add(i)

    # Find connected overlap groups using depth-first search
    visited = set()
    n_groups = 0

    for start in range(n):
        if start in visited:
            continue
        n_groups += 1
        stack = [start]
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            for neighbor in overlap_map[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return n_groups

def measure_lumin(aperture, N, radius_px, data, criteria):
    '''
    Measures luminosity recovered by a given aperture

    Inputs:
    aperture (tuple): aperture center 
    N (int): number of pixels N for an NxN modeled region
    radius_px (int, float): aperture radius in pixels
    data (array): NxN array of simulated data
    criteria (int or float): luminosity threshold 

    Returns measured luminosity, and aperture position if the luminosity is above the threshold
    
    '''
    #create a mask that sets all values outside of the aperture to 0
    mask = np.zeros((N, N), dtype=np.uint8) 
    circle = cv2.circle(mask, aperture, int(radius_px), 1, -1) 
    aperture_lumin = data*circle

    #check if the measurement is above the criteria. If yes, save the aperture position
    if sum(sum(aperture_lumin)) > criteria: 
        bright_aperture = aperture
    else:
        bright_aperture = (np.nan, np.nan)
    
    return sum(sum(aperture_lumin)), bright_aperture


def simulate_sne(rate, radius_pc, lumin_range, criteria, FOV = 200, N=500, dist='exp', scale_height_pc = 100, step_pc=10, A=40, alpha=-4/9, beta = -2.19, visualize=False):
    '''
    Simulates a population of supernovae/supernova remnants

    Inputs:
    rate (int or float): supernova rate of galaxy
    radius_pc (int or float): aperture radius in parsecs
    lumin_range (array): luminosity range of simulated population in units of 10^24 erg/s/Hz
    criteria (int or float): luminosity threshold 
    FOV (int or float): diameter of the field of view in parsecs
    N (int): number of pixels N for an NxN modeled region
    dist (String): Defines the spatial distribution of the sources. Options are 'exp' (exponential disk), 'gauss' (gaussian), and 'uniform', (uniform).  
    scale_height_pc (int or float): scale height for exponential disk distribution in parsecs
    step_pc (int or float): distance between aperture centers (ie density of aperture placements). Must be at least 1 pixel in size. 
    A (int or float): Scaling factor for luminosity-diameter relation 
    alpha (int or float): power law index for luminosity-diameter relation
    beta (int or float): power law index for luminosity function
    visualize (bool): if True, will generate a .png image of the simulated population

    Returns the measured luminosities, the number of non-redundant measurement above the threshold, and the SNe_pop class object
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
        plt.colorbar(label='1.4 GHz Luminosity [1e24 erg/s/Hz)')
        plt.title('Simulated SNe/SNR population')
        plt.xlabel('[parsecs]')
        plt.ylabel('[parsecs]')
        fig.savefig('./Simulated_pop.png', bbox_inches='tight')

    ### Add apertures and measure luminosities for each
    apertures = aperture_pos(N,N, int(radius_px), step_px=max(1,int(step_px)))
    measured = [measure_lumin(aperture, N, radius_px, data, criteria) for aperture in apertures]
    total_lumin = np.array([sublist[0] for sublist in measured])
    bright_apertures = np.array([sublist[1] for sublist in measured])[~np.isnan(np.array([sublist[1] for sublist in measured])).any(axis=1)]

    ### Determine how many (if any) of the apertures were overlapping, subtract those from the total count
    count = check_overlap(bright_apertures, radius_px) 

    return total_lumin, count, pop


def run_mc(iterations, n_jobs, rate, radius_pc, lumin_range, criteria, FOV = 200, N=2000, dist='exp', scale_height_pc = 100, step_pc=10, A=40, alpha=-4/9, beta = -2.19, visualize=False, savecsv=False):
    '''
    Simulates a multiple populations of supernovae/supernova remnants

    Inputs:
    iterations (int): the number of iterations to run the MC
    n_jobs (int): the number of jobs to use for parallelization
    rate (int or float): supernova rate of galaxy
    radius_pc (int or float): aperture radius in parsecs
    lumin_range (array): luminosity range of simulated population in units of 10^24 erg/s/Hz
    criteria (int or float): luminosity threshold 
    FOV (int or float): diameter of the field of view in parsecs
    N (int): number of pixels N for an NxN modeled region
    dist (String): Defines the spatial distribution of the sources. Options are 'exp' (exponential disk), 'gauss' (gaussian), and 'uniform', (uniform).  
    scale_height_pc (int or float): scale height for exponential disk distribution in parsecs
    step_pc (int or float): distance between aperture centers (ie density of aperture placements). Must be at least 1 pixel in size. 
    A (int or float): Scaling factor for luminosity-diameter relation 
    alpha (int or float): power law index for luminosity-diameter relation
    beta (int or float): power law index for luminosity function
    visualize (bool): if True, will generate a .png image of the simulated population

    Returns arrays of the measured luminosities, the number of non-redundant measurement above the threshold, and the SNe_pop class object
    '''
    results = Parallel(n_jobs=n_jobs)(delayed(simulate_sne)(rate=rate, radius_pc = radius_pc, lumin_range=lumin_range, criteria=criteria, FOV=FOV, N=N, dist='exp', scale_height_pc=scale_height_pc, step_pc=step_pc, A=A, alpha=alpha, beta=beta, visualize=visualize) for x in range(iterations))
    
    total_luminosities = [item[0] for item in results]
    count = [item[1] for item in results]
    population = [item[2] for item in results]

    if savecsv:
        with open('output.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Wrap each item in its own list to make it a row
            for num in count:
                writer.writerow([num])

    total_luminosities_cat = np.concatenate(total_luminosities)
    return total_luminosities_cat, count, population

        