import math as m
import itertools as it 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
from sklearn import model_selection
import time
import ray
from scipy.sparse import csr_matrix
import pickle
def inicializace_dat():
    Z = np.array([
                    3, 4, 5, 6, 7, 8, 9, 
                    11, 12, 13, 14, 15, 16, 
                    17, 19, 20, 29, 30, 31, 32, 
                    33, 34, 35, 37, 38, 47, 48, 
                    49, 50, 51, 52, 53, 55, 56
    ])
    # atomove cislo prvku A, 34 hodnot
    Prvky = np.array([
                 'Li', 'Be', 'B ', 'C ', 'N ',
                 'O ', 'F ', 'Na', 'Mg', 'Al', 'Si',
                 'P ', 'S ', 'Cl', 'K ', 'Ca', 'Cu',
                 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br',
                 'Rb', 'Sr', 'Ag', 'Cd', 'In', 'Sn',
                 'Sb', 'Te', 'I ', 'Cs', 'Ba'
    ])
    # prvky prislusejici danemu atomovemu cislu, 34 hodnot
    IP = np.array([
                  -5.329, -9.459, -8.190, -10.852, -13.585,
                  -16.433, -19.404, -5.223, -8.037,
                  -5.780, -7.758, -9.751, -11.795,
                  -13.902, -4.433, -6.428, -8.389, -10.136,
                  -5.818, -7.567, -9.262, -10.946, -12.650,
                  -4.289, -6.032, -8.058, -9.581, -5.537,
                  -7.043, -8.468, -9.867, -11.257, -4.006,
                  -5.516
    ])
    # ionizacni potencial (IP)
    EA =  np.array([
        -0.698, 0.631, -0.107, -0.872, -1.867, -3.006, -4.273, -0.716,
        0.693, -0.313, -0.993, -1.920, -2.845, -3.971, -0.621, 0.304,
        -1.638, 1.081, -0.108, -0.949, -1.839, -2.751, -3.739, -0.590,
        0.343, -1.667, 0.839, -0.256, -1.039, -1.847, -2.666, -3.513,
        -0.570, 0.278
    ])
    # elektronova afinita (EA)
    EN = np.array([
        3.014, 4.414, 4.149, 5.862, 7.726, 9.720, 11.839, 2.969, 3.672,
        3.046, 4.375, 5.835, 7.320, 8.936, 2.527, 3.062, 5.014, 4.527,
        2.963, 4.258, 5.551, 6.848, 8.194, 2.440, 2.844, 4.862, 4.371,
        2.897, 4.041, 5.158, 6.266, 7.385, 2.288, 2.619
    ])
    # Elektronegativita dle Mullikenovy definice (EN)
    HOMO = np.array([
        -2.874, -5.600, -3.715, -5.416, -7.239, -9.197, -11.294, -2.819,
        -4.782, -2.784, -4.163, -5.596, -7.106, -8.700, -2.426, -3.864,
        -4.856, -6.217, -2.732, -4.046, -5.341, -6.654, -8.001, -2.360,
        -3.641, -4.710, -5.952, -2.697, -3.866, -4.991, -6.109, -7.236,
        -2.220, -3.346
    ])

    LUMO = np.array([
        -0.978, -2.098, 2.248, 1.992, 3.057, 2.541, 1.251, -0.718, -1.358,
        0.695, 0.440, 0.183, 0.642, 0.574, -0.697, -2.133, -0.641, -1.194,
        0.130, 2.175, 0.064, 1.316, 0.708, -0.705, -1.379, -0.479, -1.309,
        0.368, 0.008, 0.105, 0.099, 0.213, -0.548, -2.129
    ])


    # The radii at which the radial probability density of the valence s, p, 
    # and d orbital are respectively maximal.
    r_s = np.array([
        1.652, 1.078, 0.805, 0.644, 0.539, 0.462, 0.406, 1.715, 1.330, 1.092,
        0.938, 0.826, 0.742, 0.679, 2.128, 1.757, 1.197, 1.099, 0.994, 0.917,
        0.847, 0.798, 0.749, 2.240, 1.911, 1.316, 1.232, 1.134, 1.057, 1.001,
        0.945, 0.896, 2.464, 2.149
    ])

    r_p = np.array([
        1.995, 1.211, 0.826, 0.630, 0.511, 0.427, 0.371, 2.597, 1.897, 1.393,
        1.134, 0.966, 0.847, 0.756, 2.443, 2.324, 1.680, 1.547, 1.330, 1.162,
        1.043, 0.952, 0.882, 3.199, 2.548, 1.883, 1.736, 1.498, 1.344, 1.232,
        1.141, 1.071, 3.164, 2.632
    ])

    r_d = np.array([
        6.930, 2.877, 1.946, 1.631, 1.540, 2.219, 1.428, 6.566, 3.171, 1.939,
        1.890, 1.771, 2.366, 1.666, 1.785, 0.679, 2.576, 2.254, 2.163, 2.373,
        2.023, 2.177, 1.869, 1.960, 1.204, 2.968, 2.604, 3.108, 2.030, 2.065,
        1.827, 1.722, 1.974, 1.351
    ])


    dE = np.array([
        -0.059, -0.038, -0.033, -0.022, 0.430, 0.506, 0.495, 0.466, 1.713,
        1.020, 0.879, 2.638, -0.146, -0.133, -0.127, -0.115, -0.178, -0.087,
        -0.055, -0.005, 0.072, 0.219, 0.212, 0.150, 0.668, 0.275, -0.146,
        -0.165, -0.166, -0.168, -0.266, -0.369, -0.361, -0.350, -0.019,
        0.156, 0.152, 0.203, 0.102, 0.275, 0.259, 0.241, 0.433, 0.341, 0.271,
        0.158, 0.202, -0.136, -0.161, -0.164, -0.169, -0.221, -0.369, -0.375,
        -0.381, -0.156, -0.044, -0.030, 0.037, -0.087, 0.070, 0.083, 0.113,
        0.150, 0.170, 0.122, 0.080, 0.016, 0.581, -0.112, -0.152, -0.158,
        -0.165, -0.095, -0.326, -0.350, -0.381, 0.808, 0.450, 0.264, 0.136,
        0.087
    ])
    # dE = E(RS) - E(ZB) ... 82 hodnot pro binární sloučeniny
    AB = np.array([
        'Li-F ', 'Li-Cl', 'Li-Br', 'Li-I ', 'Be-O ', 'Be-S ', 'Be-Se', 'Be-Te',
        'B -N ', 'B -P ', 'B -As', 'C -C ', 'Na-F ', 'Na-Cl', 'Na-Br', 'Na-I ',
        'Mg-O ', 'Mg-S ', 'Mg-Se', 'Mg-Te', 'Al-N ', 'Al-P ', 'Al-As', 'Al-Sb',
        'Si-C ', 'Si-Si', 'K -F ', 'K -Cl', 'K -Br', 'K -I ', 'Ca-O ', 'Ca-S ',
        'Ca-Se', 'Ca-Te', 'Cu-F ', 'Cu-Cl', 'Cu-Br', 'Cu-I ', 'Zn-O ', 'Zn-S ',
        'Zn-Se', 'Zn-Te', 'Ga-N ', 'Ga-P ', 'Ga-As', 'Ga-Sb', 'Ge-Ge', 'Rb-F ',
        'Rb-Cl', 'Rb-Br', 'Rb-I ', 'Sr-O ', 'Sr-S ', 'Sr-Se', 'Sr-Te', 'Ag-F ',
        'Ag-Cl', 'Ag-Br', 'Ag-I ', 'Cd-O ', 'Cd-S ', 'Cd-Se', 'Cd-Te', 'In-N ',
        'In-P ', 'In-As', 'In-Sb', 'Sn-Sn', 'B -Sb', 'Cs-F ', 'Cs-Cl', 'Cs-Br',
        'Cs-I ', 'Ba-O ', 'Ba-S ', 'Ba-Se', 'Ba-Te', 'Ge-C ', 'Sn-C ', 'Ge-Si',
        'Sn-Si', 'Sn-Ge'
    ])
    # Z vektorů dat vytvoření dictionary obsahující listy, které mají prvky svoje vypočtené hodnoty
    # Kodovani pomoci stringu nazev prvku (dva charaktery dlouhy!!!!)
    oniers = {}
    for i in range(len(Prvky)):
        oniers[Prvky[i]]= [Z[i], IP[i], EA[i], EN[i], HOMO[i], LUMO[i], r_s[i], r_p[i], r_d[i]]

    # Data jednotlivych dimeru ulozenych v dictionary listů, celkem 82 listů
    # Tyto listy jsou vlastně matice o 8 radcich a dvou sloupcich
    dimers = {} # inicializace
    temp = [] #inicializace temporary listu
    for i in AB: # pro kazdy dimer
        for j in range(1,9): # vytvori list listů osmi dvojic hodnot (bez Z[i])
            temp.append( [ oniers[i[:2]][j] , oniers[i[3:]][j] ] )
            # [IP, EA, EN, HOMO, LUMO, r_s, r_p, r_d]
        dimers[i] = temp # kodovani pomoci nazvu dimeru
        temp = [] # clearing pro dalsi iteraci
    dE = dE.reshape(-1,1) # restrukturalizace dat
    return dimers, AB, dE
    
def feature_space_generation(noise, noised_feature, sigma, dimers, AB, tier0, tier1, tier2, tier3, tier4, tier5):    
    # Nyni definujeme a nagenerujeme mnoziny moznych deskriptoru
    # Jednotlive mnoziny jsou listy floatovych hodnot ulozene jako dictionary a klic je  nazev dimeru ve formatu '__-__'
    # Brute force definice zakladnich mnozin deskriptorů pro kazdy dimer:
    
    # tier - which tier of the descriptor to include
    # Vektor popisující tvar deskriptorů:
    A1 = {}
    A2 = {}
    A3 = {}
    for i in AB:
        A1[i] = [ dimers[i][0][0] , dimers[i][1][0] , dimers[i][0][1] , dimers[i][1][1] ] 
        A2[i] = [ dimers[i][3][0] , dimers[i][4][0] , dimers[i][3][1] , dimers[i][4][1] ]
        A3[i] = [ dimers[i][5][0] , dimers[i][6][0] , dimers[i][7][0] , dimers[i][5][1] , dimers[i][6][1] , dimers[i][7][1] ]
        
    if noise==True:
        gauss = np.random.normal(1, sigma, 1)[0]
        
        if noised_feature==1 or noised_feature==True:
            for i in AB:
                A1[i][0] = A1[i][0]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==2 or noised_feature==True:
            for i in AB:
                A1[i][1] = A1[i][1]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==3 or noised_feature==True:
            for i in AB:
                A1[i][2] = A1[i][2]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==4 or noised_feature==True:
            for i in AB:
                A1[i][3] = A1[i][3]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==5 or noised_feature==True:
            for i in AB:
                A2[i][0] = A2[i][0]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==6 or noised_feature==True:
            for i in AB:
                A2[i][1] = A2[i][1]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==7 or noised_feature==True:
            for i in AB:
                A2[i][2] = A2[i][2]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==8 or noised_feature==True:
            for i in AB:
                A2[i][3] = A2[i][3]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==9 or noised_feature==True:
            for i in AB:
                A3[i][0] = A3[i][0]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==10 or noised_feature==True:
            for i in AB:
                A3[i][1] = A3[i][1]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==11 or noised_feature==True:
            for i in AB:
                A3[i][2] = A3[i][2]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==12 or noised_feature==True:
            for i in AB:
                A3[i][3] = A3[i][3]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==13 or noised_feature==True:
            for i in AB:
                A3[i][4] = A3[i][4]*np.random.normal(1, sigma, 1)[0]
        if noised_feature==14 or noised_feature==True:
            for i in AB:
                A3[i][5] = A3[i][5]*np.random.normal(1, sigma, 1)[0]
            
        
    DD = []
    DD.append('IP(A)')
    DD.append('EA(A)')
    DD.append('IP(B)')
    DD.append('EA(B)')

    DD.append('H(A)')
    DD.append('L(A)')
    DD.append('H(B)')
    DD.append('L(B)')

    DD.append('r_s(A)')
    DD.append('r_p(A)')
    DD.append('r_d(A)')
    DD.append('r_s(B)')
    DD.append('r_p(B)')
    DD.append('r_d(B)')
    
    if tier0==True:####
        DD_A1=DD[:4]
        DD_A2=DD[4:8]
        DD_A3=DD[8:14]

    # Generovani jednoduchych deskriptoru
    DD_B1 = []
    DD_B2 = []
    DD_B3 = []

    DD_C3 = []
    DD_D3 = []
    DD_E3 = []
    
    if tier1==True:####
        DD_dvojice = list( it.combinations( DD_A1 , 2 ) )
        for j in DD_dvojice:
            DD_B1.append('|'+j[0]+'-'+j[1]+'|')
            DD_B1.append('|'+j[0]+'+'+j[1]+'|')

        DD_dvojice = list( it.combinations( DD_A2 , 2 ) )
        for j in DD_dvojice:
            DD_B2.append('|'+j[0]+'-'+j[1]+'|')
            DD_B2.append('|'+j[0]+'+'+j[1]+'|')

        DD_dvojice = list( it.combinations( DD_A3 , 2 ) )
        for j in DD_dvojice:
            DD_B3.append('|'+j[0]+'-'+j[1]+'|')
            DD_B3.append('|'+j[0]+'+'+j[1]+'|')

    DD = DD + DD_B1 + DD_B2 + DD_B3
    
    if tier1==True:####
        for j in DD_A3:
            DD_C3.append(j+'^2')
            
    if tier2==True:####
        DD_dvojice = list( it.combinations( DD_A3 , 2 ) )
        for j in DD_dvojice:
            DD_C3.append('('+j[0]+'+'+j[1]+')^2')
    
    if tier1==True:####
        for j in DD_A3:
            DD_D3.append('exp('+j+')')
            
    if tier2==True:####
        DD_dvojice = list( it.combinations( DD_A3 , 2 ) )
        for j in DD_dvojice:
            DD_D3.append('exp('+j[0]+'+'+j[1]+')')
            
    if tier2==True:####
        for j in DD_A3:
            DD_E3.append('exp('+j+')^2')

    if tier3==True:
        DD_dvojice = list( it.combinations( DD_A3 , 2 ) )
        for j in DD_dvojice:
            DD_E3.append('exp('+j[0]+'+'+j[1]+')^2')

    DD = DD + DD_C3 + DD_D3 + DD_E3


    B1 = {}
    B2 = {}
    B3 = {}
    C3 = {}
    D3 = {}
    E3 = {}
    temp = []

    for i in AB:

        if tier1==True:####
            dvojice = list( it.combinations( A1[i] , 2 ) )
            for j in dvojice:
                temp.append( abs( j[0] - j[1] ) )
                temp.append( abs( j[0] + j[1] ) )

            B1[i] = temp
            temp = []


            dvojice = list( it.combinations( A2[i] , 2 ) )
            for j in dvojice:
                temp.append( abs( j[0] - j[1] ) )
                temp.append( abs( j[0] + j[1] ) )

            B2[i] = temp
            temp = []


            dvojice=list( it.combinations( A3[i] , 2 ) )
            for j in dvojice:
                temp.append( abs( j[0] - j[1] ) )
                temp.append( abs( j[0] + j[1] ) )

            B3[i] = temp
            temp = []
            
        else:
            
            B1[i] = []
            B2[i] = []
            B3[i] = []
            temp = []



        if tier1==True:####
            for j in A3[i]:
                temp.append( (j)**2 )

        if tier2==True:####
            dvojice=list( it.combinations( A3[i] , 2 ) )
            for j in dvojice:
                temp.append( ( j[0] + j[1] )**2 )

        C3[i] = temp
        temp = []

        if tier1==True:####
            for j in A3[i]:
                temp.append( m.exp( j ) )

        if tier2==True:####
            dvojice = list( it.combinations( A3[i] , 2 ) )
            for j in dvojice:
                temp.append( m.exp( j[0] + j[1] ) )

        D3[i] = temp
        temp = []

        if tier2==True:####
            for j in A3[i]:
                temp.append( m.exp( (j)**2 ) )

        if tier3==True:####
            dvojice = list( it.combinations( A3[i] , 2 ) )
            for j in dvojice:
                temp.append( m.exp( ( j[0] + j[1] )**2 ) )

        E3[i] = temp
        temp = []



    G = {}
    temp = []

    DD_G = []


    # A1,B1 ; A2,B2 lomeno A3,C3,D3,E3
    if tier1==True and tier2==False:####                    
        for j in [DD_A1, DD_A2]:
            for l in [DD_A3]:
                for k in list(it.product(j,l)):
                    DD_G.append(k[0]+'/'+k[1])

    elif tier1==True and tier2==True and tier3==False:                    
        for j in [DD_A1, DD_A2, DD_B1, DD_B2]:
            for l in [DD_A3]:
                for k in list(it.product(j,l)):
                    DD_G.append(k[0]+'/'+k[1])
                    
        for j in [DD_A1, DD_A2]:
            for l in [DD_C3[:6], DD_D3[:6]]:
                for k in list(it.product(j,l)):
                    DD_G.append(k[0]+'/'+k[1])

    elif tier1==True and tier2==True and tier3==True and tier4==False:            
        for j in [DD_A1, DD_A2, DD_B1, DD_B2]:
            for l in [DD_A3, DD_C3[:6], DD_D3[:6]]:
                for k in list(it.product(j,l)):
                    DD_G.append(k[0]+'/'+k[1])
                    
        for j in [DD_A1, DD_A2]:
            for l in [DD_C3[6:], DD_D3[6:]]:
                for k in list(it.product(j,l)):
                    DD_G.append(k[0]+'/'+k[1])

    elif tier1==True and tier2==True and tier3==True and tier4==True and tier5==False:                    
        for j in [DD_A1, DD_A2, DD_B1, DD_B2]:
            for l in [DD_A3, DD_C3, DD_D3]:
                for k in list(it.product(j,l)):
                    DD_G.append(k[0]+'/'+k[1])

        for j in [DD_A1, DD_A2, DD_B1, DD_B2]:
            for l in [DD_E3[:6]]:
                for k in list(it.product(j,l)):
                    DD_G.append(k[0]+'/'+k[1])
                    
        for j in [DD_A1, DD_A2]:
            for l in [DD_E3[6:]]:
                for k in list(it.product(j,l)):
                    DD_G.append(k[0]+'/'+k[1])

    elif tier1==True and tier2==True and tier3==True and tier4==True and tier5==True:                       
        for j in [DD_A1, DD_A2, DD_B1, DD_B2]:
            for l in [DD_A3, DD_C3, DD_D3, DD_E3]:
                for k in list(it.product(j,l)):
                    DD_G.append(k[0]+'/'+k[1])

# no tiers:
#    for j in [DD_A1, DD_A2, DD_B1, DD_B2]:
#        for l in [DD_C3, DD_D3, DD_E3]:
#            for k in list(it.product(j,l)):
#                DD_G.append(k[0]+'/'+k[1])
                    

    
    
    
    # A3/D3 a A3/E3
    if tier1==True and tier2==True and tier3==False:
        for j in [DD_D3[:6]]:
            for k in list(it.product(DD_A3,j)):
                DD_G.append(k[0]+'/'+k[1])

    elif tier1==True and tier2==True and tier3==True and tier4==False:                
        for j in [DD_D3, DD_E3[:6]]:
            for k in list(it.product(DD_A3,j)):
                DD_G.append(k[0]+'/'+k[1])

    elif tier1==True and tier2==True and tier3==True and tier4==True:
        for j in [DD_D3, DD_E3]:
            for k in list(it.product(DD_A3,j)):
                DD_G.append(k[0]+'/'+k[1])

    # B3/D3 a B3/E3
    if tier1==True and tier2==True and tier3==True and tier4==False:
        for j in [DD_D3[:6]]:
            for k in list(it.product(DD_B3,j)):
                DD_G.append(k[0]+'/'+k[1])

    elif tier1==True and tier2==True and tier3==True and tier4==True and tier5==False:
        for j in [DD_D3, DD_E3[:6]]:
            for k in list(it.product(DD_B3,j)):
                DD_G.append(k[0]+'/'+k[1])

    elif tier1==True and tier2==True and tier3==True and tier4==True and tier5==True:  
        for j in [DD_D3, DD_E3]:
            for k in list(it.product(DD_B3,j)):
                DD_G.append(k[0]+'/'+k[1])
    
    
# no tiers:
#    for j in [DD_D3, DD_E3]:
#        for k in list(it.product(DD_A3,j)):
#            DD_G.append(k[0]+'/'+k[1])
#
#    for j in [DD_D3, DD_E3]:
#        for k in list(it.product(DD_B3,j)):
#            DD_G.append(k[0]+'/'+k[1])


    # Problemove:

    # A3/A3
    if tier1==True:####
        DD_dvojice=list( it.combinations( DD_A3 , 2 ) )
        for j in DD_dvojice:
            DD_G.append(j[0]+'/'+j[1])
            DD_G.append(j[1]+'/'+j[0])


    # A3/C3
    if tier2==True:####
        DD_dvojice=list( it.combinations( DD_A3 , 2 ) )
        for j in DD_dvojice:
            DD_G.append(j[0]+'/'+'('+j[1]+')'+'^2')
            DD_G.append(j[1]+'/'+'('+j[0]+')'+'^2')
            #DD_G.append(j[0]+'/'+'('+j[0] +'+'+ j[1]+')^2')
            #DD_G.append(j[1]+'/'+'('+j[0] +'+'+ j[1]+')^2')

    if tier3==True:####
        DD_dvojice=list( it.combinations( DD_A3 , 2 ) )
        for j in DD_dvojice:
            #DD_G.append(j[0]+'/'+'('+j[1]+')'+'^2')
            #DD_G.append(j[1]+'/'+'('+j[0]+')'+'^2')
            DD_G.append(j[0]+'/'+'('+j[0] +'+'+ j[1]+')^2')
            DD_G.append(j[1]+'/'+'('+j[0] +'+'+ j[1]+')^2')


    if tier3==True:####
        DD_trojice=list(it.combinations(DD_A3,3))
        for j in DD_trojice:
            for k in [(0,1,2),(1,0,2),(2,0,1)]:
                DD_G.append(j[k[0]]+'/'+'('+j[k[1]] +'+'+ j[k[2]]+')^2')



    # B3/A3:
    if tier2==True:####
        DD_trojice=list(it.combinations(DD_A3,3))
        for j in DD_trojice:
            for k in [(0,1,2),(2,1,0),(0,2,1)]:
                DD_G.append('|'+j[k[0]] +'-'+ j[k[1]]+'| /'+j[k[2]])
    
    if tier2==True:####
        DD_dvojice=list(it.combinations(DD_A3,2))
        for j in DD_dvojice:
            DD_G.append('|'+j[1]+'-'+j[0]+'| /' + j[1])
            DD_G.append('|'+j[0]+'-'+j[1]+'| /' + j[0])

    # B3/C3
    if tier3==True:####
        DD_trojice=list(it.combinations(DD_A3,3))
        for j in DD_trojice:
            for k in [(0,1,2),(2,1,0),(0,2,1)]:
                DD_G.append('|' + j[k[0]] + '-' +  j[k[1]]+'| /'+j[k[2]]+ '^2')
    
    if tier4==True:####
        DD_dvojice=list(it.combinations(DD_A3,2))
        for j in DD_dvojice:
            DD_G.append('|'+j[0]+'-'+j[1]+'| /'+'('+j[0]+'+'+j[1]+')^2')

    if tier3==True:####
        DD_dvojice=list(it.combinations(DD_A3,2))
        for j in DD_dvojice:
            DD_G.append('|'+j[0]+'-'+j[1]+'| /' + j[0]+'^2')
            DD_G.append('|'+j[1]+'-'+j[0]+'| /'+j[1]+'^2')

    if tier4==True:####
        DD_trojice=list(it.combinations(DD_A3,3))
        for j in DD_trojice:
            DD_G.append('|'+j[0]+'-'+j[1]+'| /'+ '('+j[0]+'+'+j[2]+')^2')
            DD_G.append('|'+j[0]+'-'+j[2]+'| /'+ '('+j[0]+'+'+j[1]+')^2')

    if tier4==True:####
        DD_ctverice = list(it.combinations(DD_A3,4))
        for j in DD_ctverice:
            for k in [(0,1,2,3),(0,2,1,3),(0,3,1,2),(2,1,0,3),(3,1,0,2),(2,3,0,1)]:
                #temp.append(abs(j[k[0]]+j[k[1]])/(j[k[2]]+j[k[3]])**2)
                DD_G.append('|'+j[k[0]]+'-'+j[k[1]]+') /'+'('+j[k[2]]+'+'+j[k[3]]+')^2')


    # Zde je TEST: Ciste podily 1/r navic:
    if tier1==True:####
        for j in DD_A3:
            DD_G.append('1/' + j)

    DD =  DD + DD_G

#####################################################
    temp = []
    for i in AB:
        # A1,B1 ; A2,B2 lomeno A3,C3,D3,E3
        if tier1==True and tier2==False:####
            for j in [A1[i], A2[i]]:
                for l in [A3[i]]:
                    for k in list(it.product(j,l)):
                        temp.append(k[0]/k[1])
                        
        elif tier1==True and tier2==True and tier3==False:
            for j in [A1[i], A2[i], B1[i], B2[i]]:
                for l in [A3[i]]:
                    for k in list(it.product(j,l)):
                        temp.append(k[0]/k[1])
                        
            for j in [A1[i], A2[i]]:
                for l in [C3[i][:6], D3[i][:6]]:
                    for k in list(it.product(j,l)):
                        temp.append(k[0]/k[1])
                        
        elif tier1==True and tier2==True and tier3==True and tier4==False:            
            for j in [A1[i], A2[i], B1[i], B2[i]]:
                for l in [A3[i], C3[i][:6], D3[i][:6]]:
                    for k in list(it.product(j,l)):
                        temp.append(k[0]/k[1])
                        
            for j in [A1[i], A2[i]]:
                for l in [C3[i][6:], D3[i][6:]]:
                    for k in list(it.product(j,l)):
                        temp.append(k[0]/k[1])
                        
        elif tier1==True and tier2==True and tier3==True and tier4==True and tier5==False:
            for j in [A1[i], A2[i], B1[i], B2[i]]:
                for l in [A3[i], C3[i], D3[i]]:
                    for k in list(it.product(j,l)):
                        temp.append(k[0]/k[1])
                        
            for j in [A1[i], A2[i], B1[i], B2[i]]:
                for l in [E3[i][:6]]:
                    for k in list(it.product(j,l)):
                        temp.append(k[0]/k[1])
                        
            for j in [A1[i], A2[i]]:
                for l in [E3[i][6:]]:
                    for k in list(it.product(j,l)):
                        temp.append(k[0]/k[1])

        elif tier1==True and tier2==True and tier3==True and tier4==True and tier5==True:   
            for j in [A1[i], A2[i], B1[i], B2[i]]:
                for l in [A3[i], C3[i], D3[i], E3[i]]:
                    for k in list(it.product(j,l)):
                        temp.append(k[0]/k[1])
###################################################
                        
                        
###################################################
        # A3/D3 a A3/E3
        if tier1==True and tier2==True and tier3==False:
            for j in [D3[i][:6]]:
                for k in list(it.product(A3[i],j)):
                    temp.append(k[0]/k[1])
                    
        elif tier1==True and tier2==True and tier3==True and tier4==False:
            for j in [D3[i], E3[i][:6]]:
                for k in list(it.product(A3[i],j)):
                    temp.append(k[0]/k[1])
                
        elif tier1==True and tier2==True and tier3==True and tier4==True:
            for j in [D3[i], E3[i]]:
                for k in list(it.product(A3[i],j)):
                    temp.append(k[0]/k[1])
                
        # B3/D3 a B3/E3
        if tier1==True and tier2==True and tier3==True and tier4==False:
            for j in [D3[i][:6]]:
                for k in list(it.product(B3[i],j)):
                    temp.append(k[0]/k[1])
                    
        elif tier1==True and tier2==True and tier3==True and tier4==True and tier5==False:
            for j in [D3[i], E3[i][:6]]:
                for k in list(it.product(B3[i],j)):
                    temp.append(k[0]/k[1])
                    
        elif tier1==True and tier2==True and tier3==True and tier4==True and tier5==True:  
            for j in [D3[i], E3[i]]:
                for k in list(it.product(B3[i],j)):
                    temp.append(k[0]/k[1])

####################################################
                    
        # PROBLEMOVE PODILY:

        # A3/A3 - vsechny podily jenom ne podily X/X = 1:
        if tier1==True:####
            dvojice=list( it.combinations( A3[i] , 2 ) )
            for j in dvojice:
                temp.append(j[0]/j[1])
                temp.append(j[1]/j[0])


        # A3/C3:
        if tier2==True:####
            dvojice=list(it.combinations(A3[i],2))
            for j in dvojice:
                temp.append(j[0]/(j[1])**2)
                temp.append(j[1]/(j[0])**2)
                #temp.append(j[0]/(j[0] + j[1])**2)
                #temp.append(j[1]/(j[0] + j[1])**2)
        
        if tier3==True:####
            dvojice=list(it.combinations(A3[i],2))
            for j in dvojice:
                #temp.append(j[0]/(j[1])**2)
                #temp.append(j[1]/(j[0])**2)
                temp.append(j[0]/(j[0] + j[1])**2)
                temp.append(j[1]/(j[0] + j[1])**2)                
                
                
        if tier3==True:
            trojice=list(it.combinations(A3[i],3))
            for j in trojice:
                for k in [(0,1,2),(1,0,2),(2,0,1)]:
                    temp.append(j[k[0]]/(j[k[1]] + j[k[2]])**2)


        # B3/A3:
        if tier2==True:
            trojice=list(it.combinations(A3[i],3))
            for j in trojice:
                for k in [(0,1,2),(2,1,0),(0,2,1)]:
                    temp.append(abs(j[k[0]] - j[k[1]])/j[k[2]])

        if tier2==True:
            dvojice=list(it.combinations(A3[i],2))
            for j in dvojice:
                temp.append(abs(1-(j[0]/j[1])))
                temp.append(abs(1-(j[1]/j[0])))


        # B3/C3
        if tier3==True:
            trojice=list(it.combinations(A3[i],3))
            for j in trojice:
                for k in [(0,1,2),(2,1,0),(0,2,1)]:
                    temp.append(abs(j[k[0]] - j[k[1]])/j[k[2]]**2)

        if tier4==True:
            dvojice=list(it.combinations(A3[i],2))
            for j in dvojice:
                temp.append(abs(j[0]-j[1])/(j[0]+j[1])**2)

        if tier3==True:
            dvojice=list(it.combinations(A3[i],2))
            for j in dvojice:
                temp.append(abs(j[0]-j[1])/j[0]**2)
                temp.append(abs(j[1]-j[0])/j[1]**2)

        if tier4==True:
            trojice=list(it.combinations(A3[i],3))
            for j in trojice:
                temp.append(abs(j[0]-j[1])/(j[0]+j[2])**2)
                temp.append(abs(j[0]-j[2])/(j[0]+j[1])**2)

        if tier4==True:
            ctverice = list(it.combinations(A3[i],4))
            for j in ctverice:
                for k in [(0,1,2,3),(0,2,1,3),(0,3,1,2),(2,1,0,3),(3,1,0,2),(2,3,0,1)]:
                    #temp.append(abs(j[k[0]]+j[k[1]])/(j[k[2]]+j[k[3]])**2)
                    temp.append(abs(j[k[0]]-j[k[1]])/(j[k[2]]+j[k[3]])**2)






        # Zde je TEST: Ciste podily 1/r navic:
        if tier1==True:####
            for j in A3[i]:
                temp.append(1/j)


        G[i] = temp
        temp = []



    # F1, F2, F3: ... 44 deskriptoru celkem
    if tier3==True:####
        DD_F1 = []
        for j in list(it.combinations(DD_A1[:2],2)):
            for k in list(it.combinations(DD_A1[2:],2)):
                DD_F1.append( "|" + "|" + j[0] + "-" + j[1] + "|" + "+" + "|" + k[0] + "-" + k[1] + "|" + "|" )
                DD_F1.append( "|" + "|" + j[0] + "-" + j[1] + "|" + "-" + "|" + k[0] + "-" + k[1] + "|" + "|" )
                DD_F1.append( "|" + "|" + j[0] + "+" + j[1] + "|" + "+" + "|" + k[0] + "+" + k[1] + "|" + "|" )
                DD_F1.append( "|" + "|" + j[0] + "+" + j[1] + "|" + "-" + "|" + k[0] + "+" + k[1] + "|" + "|" )

        DD = DD + DD_F1

        DD_F2 = []
        for j in list(it.combinations(DD_A2[:2],2)):
            for k in list(it.combinations(DD_A2[2:],2)):
                DD_F2.append( "|" + "|" + j[0] + "-" + j[1] + "|" + "+" + "|" + k[0] + "-" + k[1] + "|" + "|" )
                DD_F2.append( "|" + "|" + j[0] + "-" + j[1] + "|" + "-" + "|" + k[0] + "-" + k[1] + "|" + "|" )
                DD_F2.append( "|" + "|" + j[0] + "+" + j[1] + "|" + "+" + "|" + k[0] + "+" + k[1] + "|" + "|" )
                DD_F2.append( "|" + "|" + j[0] + "+" + j[1] + "|" + "-" + "|" + k[0] + "+" + k[1] + "|" + "|" )

        DD = DD + DD_F2

        DD_F3 = []
        for j in list(it.combinations(DD_A3[:3],2)):
            for k in list(it.combinations(DD_A3[3:],2)):
                DD_F3.append( "|" + "|" + j[0] + "-" + j[1] + "|" + "+" + "|" + k[0] + "-" + k[1] + "|" + "|" )
                DD_F3.append( "|" + "|" + j[0] + "-" + j[1] + "|" + "-" + "|" + k[0] + "-" + k[1] + "|" + "|" )
                DD_F3.append( "|" + "|" + j[0] + "+" + j[1] + "|" + "+" + "|" + k[0] + "+" + k[1] + "|" + "|" )
                DD_F3.append( "|" + "|" + j[0] + "+" + j[1] + "|" + "-" + "|" + k[0] + "+" + k[1] + "|" + "|" )
        
        DD = DD + DD_F3

    # F1, F2, F3: ... 52 deskriptoru celkem

    F1 = {}
    F2 = {}
    F3 = {}

    temp = []
    if tier3==True:####
        for i in AB:
            for j in list(it.combinations(A1[i][:2],2)):
                for k in list(it.combinations(A1[i][2:],2)):
                    temp.append( abs( abs(j[0]-j[1]) + abs(k[0]-k[1]) ) )
                    temp.append( abs( abs(j[0]-j[1]) - abs(k[0]-k[1]) ) )
                    temp.append( abs( abs(j[0]+j[1]) + abs(k[0]+k[1]) ) )
                    temp.append( abs( abs(j[0]+j[1]) - abs(k[0]+k[1]) ) )

            F1[i] = temp
            temp = []


        for i in AB:
            for j in list(it.combinations(A2[i][:2],2)):
                for k in list(it.combinations(A2[i][2:],2)):
                    temp.append( abs( abs(j[0]-j[1]) + abs(k[0]-k[1]) ) )
                    temp.append( abs( abs(j[0]-j[1]) - abs(k[0]-k[1]) ) )
                    temp.append( abs( abs(j[0]+j[1]) + abs(k[0]+k[1]) ) )
                    temp.append( abs( abs(j[0]+j[1]) - abs(k[0]+k[1]) ) )

            F2[i] = temp
            temp = []


        for i in AB:
            for j in list(it.combinations(A3[i][:3],2)):
                for k in list(it.combinations(A3[i][3:],2)):
                    temp.append( abs( abs(j[0]-j[1]) + abs(k[0]-k[1]) ) )
                    temp.append( abs( abs(j[0]-j[1]) - abs(k[0]-k[1]) ) )
                    temp.append( abs( abs(j[0]+j[1]) + abs(k[0]+k[1]) ) )
                    temp.append( abs( abs(j[0]+j[1]) - abs(k[0]+k[1]) ) )

            F3[i] = temp
            temp = []
    else:
        for i in AB:
            F1[i] = []
            F2[i] = []
            F3[i] = []
            temp = []
        
    # 
    Descriptors = len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+ len(B1[AB[0]])+len(B2[AB[0]])+len(B3[AB[0]])+len(C3[AB[0]])+len(D3[AB[0]])+len(E3[AB[0]])+len(G[AB[0]])+len(F1[AB[0]])+len(F2[AB[0]])+len(F3[AB[0]])

    # Matice D. Opustíme dictionary a použijeme np.array
    D=np.empty((82,Descriptors),dtype=float)
    for i in range(len(AB)):
        for j in range(len(A1[AB[i]])):
            D[i][j] = A1[AB[i]][j]

        for j in range(len(A2[AB[i]])):
            D[i][j+len(A1[AB[i]])] = A2[AB[i]][j]

        for j in range(len(A3[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])] = A3[AB[i]][j]

        for j in range(len(B1[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])] = B1[AB[i]][j]

        for j in range(len(B2[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])+len(B1[AB[i]])] = B2[AB[i]][j]

        for j in range(len(B3[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])+len(B1[AB[i]])+len(B2[AB[i]])] = B3[AB[i]][j]

        for j in range(len(C3[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])+len(B1[AB[i]])+len(B2[AB[i]])+len(B3[AB[i]])] = C3[AB[i]][j]

        for j in range(len(D3[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])+len(B1[AB[i]])+len(B2[AB[i]])+len(B3[AB[i]])+len(C3[AB[i]])] = D3[AB[i]][j]

        for j in range(len(E3[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])+len(B1[AB[i]])+len(B2[AB[i]])+len(B3[AB[i]])+len(C3[AB[i]])+len(D3[AB[i]])] = E3[AB[i]][j]

        for j in range(len(G[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])+len(B1[AB[i]])+len(B2[AB[i]])+len(B3[AB[i]])+len(C3[AB[i]])+len(D3[AB[i]])+len(E3[AB[i]])] = G[AB[i]][j]

        for j in range(len(F1[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])+len(B1[AB[i]])+len(B2[AB[i]])+len(B3[AB[i]])+len(C3[AB[i]])+len(D3[AB[i]])+len(E3[AB[i]])+len(G[AB[i]])] = F1[AB[i]][j]

        for j in range(len(F2[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])+len(B1[AB[i]])+len(B2[AB[i]])+len(B3[AB[i]])+len(C3[AB[i]])+len(D3[AB[i]])+len(E3[AB[i]])+len(G[AB[i]])+len(F1[AB[i]])] = F2[AB[i]][j]

        for j in range(len(F3[AB[i]])):
            D[i][j+len(A1[AB[i]])+len(A2[AB[i]])+len(A3[AB[i]])+len(B1[AB[i]])+len(B2[AB[i]])+len(B3[AB[i]])+len(C3[AB[i]])+len(D3[AB[i]])+len(E3[AB[i]])+len(G[AB[i]])+len(F1[AB[i]])+len(F2[AB[i]])] = F3[AB[i]][j]


    #print('A1: ', '0 ... ',len(A1[AB[0]])-1)
    #print('A2: ',len(A1[AB[0]]),' ... ',len(A1[AB[0]])+len(A2[AB[0]])-1)
    #print('A3: ',len(A1[AB[0]])+len(A2[AB[0]]), ' ... ', len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])-1)
    #print('B1: ',len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]]), ' ... ',  len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[i]])-1)
    #print('B2: ',len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]]), ' ... ',len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]])+len(B2[AB[0]])-1)
    #print('B3: ',len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]])+len(B2[AB[0]]), ' ... ', len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]])+len(B2[AB[0]])+len(B3[AB[0]])-1)
    #print('C3: ', len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]])+len(B2[AB[0]])+len(B3[AB[0]]),' ... ', len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]])+len(B2[AB[0]])+len(B3[AB[0]])+len(C3[AB[0]])-1)
    #print('D3: ',len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]])+len(B2[AB[0]])+len(B3[AB[0]])+len(C3[AB[0]]), ' ... ', len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]])+len(B2[AB[0]])+len(B3[AB[0]])+len(C3[AB[0]])+len(D3[AB[0]])-1)
    #print('E3: ',len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]])+len(B2[AB[0]])+len(B3[AB[0]])+len(C3[AB[0]])+len(D3[AB[0]]), ' ... ', len(A1[AB[0]])+len(A2[AB[0]])+len(A3[AB[0]])+len(B1[AB[0]])+len(B2[AB[0]])+len(B3[AB[0]])+len(C3[AB[0]])+len(D3[AB[0]])+len(E3[AB[0]])-1)
    
    return D, DD, A1, A2, A3, B1, B2, B3, C3, D3, E3, F1, F2, F3, G

def lasso_filter(D, dE, POCET, LAMBDA_LENGTH):
    #D_normalized=preprocessing.normalize(D,axis=0)
    D_normalized=preprocessing.StandardScaler().fit_transform(D)
    LAMBDA=np.empty(LAMBDA_LENGTH,dtype=float)
    LAMBDA[0]=(1/D.shape[0])*np.max((D_normalized.T).dot(dE))
    q=m.pow(1000,1/(LAMBDA_LENGTH-1))
    for i in range(1,len(LAMBDA)):
        LAMBDA[i]=LAMBDA[i-1]/q


    lassocv = linear_model.LassoCV(n_jobs=-1)#(fit_intercept=True, normalize=False, max_iter=1e4, tol=1e-3, copy_X=True)
    #alphas, coefs, _ = lassocv.path(D_normalized, dE, fit_intercept=True, alphas=None, max_iter=1e4, tol=1e-4)
    alphas, coefs, _  = linear_model.Lasso.path(D_normalized, dE, normalize=False, l1_ratio=1, alphas=LAMBDA, max_iter=1e4, tol=1e-4, n_jobs=-1)

    sparse = csr_matrix(coefs[0].T)

    THETA = set()
    for i in range(LAMBDA_LENGTH):
        for el in sparse[i].indices:
            if len(THETA)<POCET:
                THETA.add(el)

    THETA = list(THETA)
    THETA.sort()

    return THETA

def lo(THETA, D, dE):
    # Definujeme nejhorsi mozne MSE, tj. MSE pri coef = 0 a intercept = 0:
    MSE = np.ones((4),dtype = float)*metrics.mean_squared_error(dE, np.zeros(dE.shape, dtype = float)) # ctyr dimenzionalni MSE vektor
    MaxAE = np.zeros((4),dtype = float)
    ND = [ [], [], [], [] ]
    # Specialni rutina pro 1D deskriptor

    for i in THETA:
        OMEGA = D[:,i].reshape(-1, 1)
        ols = linear_model.LinearRegression(fit_intercept = True, normalize = False)
        #OMEGA = preprocessing.StandardScaler().fit_transform(OMEGA)
        ols.fit(OMEGA, dE)
        dE_predicted = ols.predict(OMEGA)
        novy_model_MSE = metrics.mean_squared_error(dE, dE_predicted)

        if novy_model_MSE < MSE[0]:
            MSE[0] = novy_model_MSE
            MaxAE[0] = metrics.max_error(dE,dE_predicted)
            ND[0] = [i, ols.coef_, ols.intercept_, m.sqrt(MSE[0]), MaxAE[0]]

    # Rutina pro 2D, ... ,4D deskriptor
    for j in range(2,5): # 2,3,4
        OMEGA = np.zeros((D.shape[0],j),dtype = float)
        jetice = it.combinations(THETA, j)

        for i in jetice:

            for k in range(j):
                OMEGA[:,k] = D[:,i[k]]

            #OMEGA = preprocessing.StandardScaler().fit_transform(OMEGA)
            ols = linear_model.LinearRegression(fit_intercept = True, normalize = False)
            ols.fit(OMEGA,dE)
            dE_predicted = ols.predict(OMEGA)
            novy_model_MSE = metrics.mean_squared_error(dE,dE_predicted)

            if novy_model_MSE < MSE[j-1]:
                MSE[j-1] = novy_model_MSE
                MaxAE[j-1] = metrics.max_error(dE,dE_predicted)
                ND[j-1] = [i, ols.coef_, ols.intercept_, m.sqrt(MSE[j-1]), MaxAE[j-1]]
    return ND

def lo_2D(THETA, D, dE):
    # Definujeme nejhorsi mozne MSE, tj. MSE pri coef = 0 a intercept = 0:
    MSE = metrics.mean_squared_error(dE,np.zeros(dE.shape,dtype = float)) # ctyr dimenzionalni MSE vektor
    MaxAE = 0
    ND = 0
    OMEGA = np.zeros((D.shape[0], 2),dtype = float)
    jetice = it.combinations(THETA, 2)

    for i in jetice:
        for k in range(2):
            OMEGA[:,k] = D[:,i[k]]
        ols = linear_model.LinearRegression(fit_intercept = True, normalize = True)
        ols.fit(OMEGA,dE)
        dE_predicted = ols.predict(OMEGA)
        novy_model_MSE = metrics.mean_squared_error(dE, dE_predicted)

        if novy_model_MSE < MSE:
            MSE = novy_model_MSE
            MaxAE = metrics.max_error(dE,dE_predicted)
            ND = [i, ols.coef_, ols.intercept_, m.sqrt(MSE), MaxAE]
    return ND
@ray.remote(num_return_vals=4)
def cross_validation_LASSO_2D(cross_iter, POCET, LAMBDA_LENGTH, D, dE, test_size=7):
    print("Starting...")
    # Vektory co budou drzet hodnoty pro kazdou cross validaci
    RMSE_CV = []#np.empty((cross_iter),dtype = float)
    MaxAE_CV = []#np.empty((cross_iter),dtype = float)
    recovery = np.zeros(cross_iter)
    cross_validation_descriptors = np.zeros(cross_iter, dtype=object)

    for cv in range(cross_iter):
        if test_size==1 and cross_iter==82: # LOOCV
            D_CV = D[[i for i in range(cross_iter) if i!=cv], :]
            X_test = np.array([D[cv, :]])
            dE_CV = dE[[i for i in range(cross_iter) if i!=cv], :]
            y_test = np.array(dE[cv, :])
        else:
            D_CV, X_test, dE_CV, y_test = model_selection.train_test_split(D, dE, test_size=test_size, random_state=cv, shuffle = True)

        THETA = lasso_filter(D_CV, dE_CV, POCET, LAMBDA_LENGTH)
        ND = lo_2D(THETA, D_CV, dE_CV)

        #Pro 2D:
        temporary = np.zeros((y_test.shape[0], 2),dtype = float)
        for k in range(2):
            temporary[:,k] = X_test[:,ND[0][k]]
        y_predicted = np.ones((y_test.shape[0],1),dtype = float)*ND[2] + np.dot(temporary,ND[1].T)
        RMSE_CV.append(np.sqrt(metrics.mean_squared_error(y_test, y_predicted)))
        MaxAE_CV.append(metrics.max_error(y_test, y_predicted))

        # column indices of descriptors
        desc = [819, (966, 2717), (966, 2717, 3110), (2151, 2717, 3110, 3399)]

        print(sorted(ND[0])==sorted(desc[1]), sorted(ND[0]), sorted(desc[1]))
        if sorted(ND[0])==sorted(desc[1]):
            recovery[cv] = 1

        cross_validation_descriptors[cv] = ND[0]
    print("Done.")
    return RMSE_CV, MaxAE_CV, recovery, cross_validation_descriptors

# LOOCV sens analysis
sigma = [0.001, 0.01, 0.03, 0.05, 0.1, 0.13, 0.3] # noise

start = time.time()
dimers, AB, dE = inicializace_dat()

POCET = 30
LAMBDA_LENGTH = 100
cross_iter = 82
iterace = 50

#sens_data = [] #np.empty((len(sigma), iterace), dtype=object) # 14

feat = 1 # 1..14 

hluk = 2 # 1 done, 2 done

ray.init(num_cpus=32)

for sig in [sigma[hluk-1]]: # for noise level
    object_ids = []
    for _ in range(iterace): # do fifty drafts
        #D_noised, DD, A1, A2, A3, B1, B2, B3, C3, D3, E3, F1, F2, F3, G = feature_space_generation(True, feat, sig, dimers, AB, True, True, True, True, True, True)

        #RMSE_CV, MaxAE_CV, recovery, cross_validation_descriptors = cross_validation_LASSO_2D.remote(cross_iter, POCET, LAMBDA_LENGTH, feature_space_generation(True, feat, sig, dimers, AB, True, True, True, True, True, True)[0], dE, test_size=1)

        object_ids.append(cross_validation_LASSO_2D.remote(cross_iter, POCET, LAMBDA_LENGTH, feature_space_generation(True, feat, sig, dimers, AB, True, True, True, True, True, True)[0], dE, test_size=1))

    sens_data = []
    for i in range(iterace): # do fifty drafts
        sens_data.append(ray.get(object_ids[i]))###
    pickle.dump( sens_data, open( "sens_data_LOOCV_prim_f_" + str(feat) + "_n_" + str(hluk) + ".p", "wb" ) )

print("Elapsed: ", (time.time() - start)/60, " min.")
ray.shutdown()
