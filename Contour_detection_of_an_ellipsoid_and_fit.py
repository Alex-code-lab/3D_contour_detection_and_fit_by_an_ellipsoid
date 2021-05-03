# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 13:51:34 2020

@author: Alexandre_Souchaud

This program has been developped in order to treat confocal images. 
Some stacks have been saved. 
To recontruct the shape in 3D, an activ contour function has been used. 
Then, this contour is fitted by an ellipsoid.
All this parameters are used to analyse the evolution of the shape in time.


The active contour method used is morphological_geodesic_active_contour
https://github.com/scikit-image/scikit-image/blob/main/skimage/segmentation/morphsnakes.py


The parameters given in this program are choosen for the pictures given in exemple

I have to add a warning : indeed the function "inverse_gaussian_gradient" is working with 
Python 3.6.7 64-bit | Qt 5.9.6 | PyQt5 5.9.2 | Windows 10 
This function might change in the latest version of python and I didn't check it out.

"""
#%%
from fonctions import *
import numpy as np
from scipy import ndimage
from skimage import io
import os
from skimage.segmentation import circle_level_set, inverse_gaussian_gradient, morphological_geodesic_active_contour
import glob
import re
import skimage
from math import ceil
#%%
# =============================================================================
# =============================================================================
# =============================================================================
#                           Main : Coeur du programme : 
# =============================================================================
# =============================================================================
# =============================================================================
# Location of the files. 
dossier_serie = os.getcwd() # dossier de base ou se trouve le programme 
path_dossier_image = dossier_serie + "/*.tif" #les images à étudier
liste_element = glob.glob(path_dossier_image) # formation de la liste des images à traiter
path_dossier_travail = dossier_serie# + "\\Traitement des agrégats complets" # enregistrement du chemin de dossier de travail   
# =============================================================================
# Creating the text file in order to save the results of the fits. 
# =============================================================================
data_path_folder = os.getcwd()
fichier = open("data.txt","a")  
fichier.write("données d'analyse : ")
fichier.write("\n"); fichier.write("\n")
column = "name volume_R1v volume_z1 Volume_Y_1 Volume_X_1 Surface_length_1 Surface_Z_1 Surface_Y_1 Surface_X_1 Volume_length_2 Volume_Z_2 Volume_Y_2 Volume_X_2 Surface_length_2 Surface_Z_2 Surface_Y_2 Surface_X_2 Volume_length_3 Volume_Z_3 Volume_Y_3 Volume_X_3 Surface_length_3 Surface_Z_3 Surface_Y_3 Surface_X_3 centre_ellipsoid_1 centre_ellipsoid_2 centre_ellipsoid_3"
fichier.write("\n"); fichier.write(column); fichier.close()
data_path_field = data_path_folder + "/data.txt"     
#%%
#Read the file. 
for element in liste_element : 
    print(element)
    image = io.imread(element)     
    if len(image.shape)>3 :
        try :
            from skimage.color import rgb2gray
            image = rgb2gray(image)
        except ImportError:
            raise ImportError("Image au mauvais format") 
    if len(image.shape)<3 :
            raise ImportError("Image au mauvais format : manque de stack") 
# =============================================================================
# For each picture in each field, we make a new field to save new pictures
# =============================================================================
#    Fields names
# =============================================================================
#    For linux
#    name = re.sub(dossier_serie," ",element)
# =============================================================================
    a = dossier_serie.replace('\\','/') 
    b = element.replace('\\','/')
    name = re.sub(a,"",b)       
    name = re.sub('/',"",name)
    name = re.sub(".tif","",name)   
    print("study of: ", name)
#   Creating a field for the specific picture
    os.mkdir(name)
    os.chdir(name) 
    dossier_image_specifique = os.getcwd()
# =============================================================================
#                       Paramètres d'analyse
# ============================================================================= 
#==========================Peut être modifié===================================
# pixels size
    size_pix_x = size_pix_y = 0.1628145
    size_pix_z_ini = 1.500130
    # Correction factor in z axis
    factor = 0.935
    size_pix_z = size_pix_z_ini*factor
    size_pix = (size_pix_z,size_pix_x,size_pix_y)
# Itération number for the contour detection
    nbr_iteration = 200; 
# Parameters 
    radius_circle = 5 # Sphere radius
    sigmas = (2,4,4) #(z,x,y)
    median = (1,3,3) #(z,x,y)
#================= A laisser ==================================================
# other fixed parameters  
    tresh = 'auto'; smooth = 1; balloonn = 1; alphas = 150;  
# =============================================================================
# =============================================================================      
# saving the pretreated pictures  
    if median is None : 
        image2 = skimage.exposure.rescale_intensity(image)    
        image2 = skimage.exposure.equalize_hist(image2)
        io.imsave('image_histo_egalise.tif', np.array(image2).astype(np.float32))
    else :    
        image2 = ndimage.filters.median_filter(image, size = None, footprint = np.ones(median), mode = 'nearest')
        image2 = skimage.exposure.rescale_intensity(image)                     
        image2 = skimage.exposure.equalize_hist(image2)
        io.imsave('image_histo_egalise_median.tif', np.array(image2).astype(np.float32))   
# Contour developped by a gaussian gradient
    gimage = inverse_gaussian_gradient(image2, alpha = alphas, sigma = sigmas)
    io.imsave('gimage.tif', np.array(gimage).astype(np.float32))   
# Center of the study
# We only keep the 2% points with the intensity max in order to fin approximatively
#the center. It's only for the pretreatment. No need to be preicse.

    thresh = np.percentile(image2,98)
    mask = (image2>thresh)*image2       
    zc,xc,yc = ndimage.measurements.center_of_mass(mask)
    if np.isnan(zc):
        zc= image2.shape[0]/2
        xc= image2.shape[1]/2
        yc= image2.shape[2]/2  
# définition de la sphere initiale
    initi_ls = circle_level_set(image.shape, center=(zc,xc,yc), radius=radius_circle)
    # Active contour method
    evolution = []
    callback = store_evolution_in(evolution)           
    ls = morphological_geodesic_active_contour(gimage, nbr_iteration, initi_ls,
                                                   smoothing = smooth, balloon= balloonn,
                                                   threshold= tresh , iter_callback = callback)#Balloon = gonfle si positif , dégonfle si negatif 
#%%  
# =============================================================================
#    # Saving the datas       
# =============================================================================
#    io.imsave("volume_{}_{}_{}.tif".format(name,median,sigmas), np.array(evolution[nbr_iteration]).astype(np.float32))
    contour = image_contour(evolution[nbr_iteration])
    io.imsave("contour_{}_{}_{}.tif".format(name,median,sigmas), contour)
    stack_volume =  np.array(evolution[nbr_iteration]).astype(np.float32)
    del evolution, ls, gimage, callback,mask,zc, xc, yc , tresh,image2
           
#%%  
# =============================================================================
# #===== fitting d'un ellipsoide sur le contour du volume =======================
# =============================================================================
# Boucle des traitements  
#     Mise en forme des données pour fitting d'un ellipsoïde
    Z,X,Y = np.where(contour>0)
    I = np.array([contour[np.where(contour>0)]]).T
    if np.NaN in I : 
        os.chdir(dossier_serie); continue;
    if np.inf in I : 
        os.chdir(dossier_serie); continue;
    if len(I) == 0 : 
        os.chdir(dossier_serie) ; continue ;
    arr = np.zeros((X.shape[0],3))
    arr[:,0] = Z*size_pix_z; arr[:,1] = X*size_pix_x; arr[:,2] = Y*size_pix_y
    if len(arr) == 0 :
        os.chdir(dossier_serie); continue;
# fit d'un ellipsoid par rapport au contour
    (a_cont,b_cont,c_cont),P_cont, C_0 = fit_ellipsoide(arr,I)         
#%% 
# =============================================================================
# #===== Fitting sur le volume avec matrice d'inertie==========================                
# =============================================================================
# Définition du centre de masse et calcul de la matrice d'inertie    
    centre_masse_z,centre_masse_x,centre_masse_y = ndimage.measurements.center_of_mass(stack_volume)
    D_vol,P_vol,masse_bille = matrix_inertia(stack_volume,centre_masse_z,centre_masse_x,centre_masse_y,(size_pix_x,size_pix_y,size_pix_z))
# Fitting d'un ellipsoïde avec volume
    a_vol, b_vol, c_vol, volume_vol = ellipsoide_parametres(D_vol, masse_bille)        
#%%  
# =============================================================================
# Calculation and generation of the fitted ellipsoid with the contour method
# =============================================================================
    ac, bc, cc, Pcont, etat_cont = matrice_passage(a_cont,b_cont,c_cont,P_cont)
    tetac, psic, phic, Pcont = angle(Pcont)
    centre = (centre_masse_x,centre_masse_y,centre_masse_z*size_pix_z/size_pix_x)
    ellipsoide_generee_3D = generate_ellipsoide(ac/size_pix_x,bc/size_pix_x,cc/size_pix_x,centre_masse_z*size_pix_z/size_pix_x,
                                                centre_masse_x,centre_masse_y,psic,tetac,phic,(int(image.shape[0]*size_pix_z_ini/size_pix_y),image.shape[1],image.shape[2])) 
    ellipsoide_genere_shape_image = np.zeros_like(image)
    i=0
    for k in range(0,ellipsoide_generee_3D.shape[0],ceil(size_pix_z_ini/size_pix_x)):
        ellipsoide_genere_shape_image[i,:,:] = ellipsoide_generee_3D[k,:,:]  
        i=i+1            
    io.imsave("ellipsoide_contour_{}_{}_{}_{}.tif".format(name,median,sigmas,factor), np.array(ellipsoide_genere_shape_image).astype(np.float32))   
# =============================================================================
#  Calculation and generation of the fitted ellipsoid with the volume method
# =============================================================================
    av, bv, cv, Pvol, etat_vol = matrice_passage(a_vol,b_vol,c_vol,P_vol)
    tetav, psiv, phiv,Pvol = angle(Pvol)
    centre = (centre_masse_x,centre_masse_y,centre_masse_z*size_pix_z/size_pix_x)
    ellipsoide_generee_3D = generate_ellipsoide(av/size_pix_x,bv/size_pix_x,cv/size_pix_x,centre_masse_z*size_pix_z/size_pix_x,
                                                centre_masse_x,centre_masse_y,psiv,tetav,phiv,(int(image.shape[0]*size_pix_z_ini/size_pix_y),image.shape[1],image.shape[2])) 
    ellipsoide_genere_shape_image = np.zeros_like(image)
    i=0
    for k in range(0,ellipsoide_generee_3D.shape[0],ceil(size_pix_z_ini/size_pix_x)):
        ellipsoide_genere_shape_image[i,:,:] = ellipsoide_generee_3D[k,:,:]  
        i=i+1
#        io.imsave("elli_vol.tif", np.array(ellipsoide_generee_3D).astype(np.float32))
    io.imsave("ellipsoide_volume_{}_{}_{}_{}.tif".format(name,median,sigmas,factor), np.array(ellipsoide_genere_shape_image).astype(np.float32))      
    P_cont = Pcont
    P_vol = Pvol
    del Pcont, Pvol, etat_cont
# =============================================================================
#         Save the datas
# =============================================================================               
    fichier = open(data_path_field,"a")         
    fichier.write("\n")      
    values = name + " " + str(av)  + " " + str(P_vol[0][0]) + " " + str(P_vol[1][0]) + " " + str(P_vol[2][0]) + " " + str(ac) + " " + str(P_cont[0][0]) + " " + str(P_cont[1][0]) + " " + str(P_cont[2][0]) + " " + str(bv)  + " " + str(P_vol[0][1]) + " " + str(P_vol[1][1]) + " " + str(P_vol[2][1]) + " " + str(bc) + " " + str(P_cont[0][1]) + " " + str(P_cont[1][1]) + " " + str(P_cont[2][1])+ " "+ str(cv)  + " " + str(P_vol[0][2]) + " " + str(P_vol[1][2]) + " " + str(P_vol[2][2]) + " " + str(bc) + " " + str(P_cont[0][2]) + " " + str(P_cont[1][2]) + " " + str(P_cont[2][2]) + " " + str(C_0[0]) + " " + str(C_0[1]) + " "+ str(C_0[2])
    fichier.write(values)
    fichier.write("\n")     
    fichier.close()  
    # Retour aux différents dossiers
    del arr, X, Y, Z, contour, P_vol, P_cont, ac, bc, cc, tetac, psic, phic, av, bv, cv, etat_vol, tetav, psiv, phiv
    del ellipsoide_genere_shape_image, ellipsoide_generee_3D, stack_volume
    os.chdir(dossier_serie)   
# =============================================================================
#  retreated the datas to be study with Excel.
# =============================================================================
fichier=open("data.txt", "r")
contenu=fichier.read() #Lit tout le fichier d'un coup
index = { '.':',', "[":'',"]":'' } 
for cle in index:
    contenu=contenu.replace(cle, index[cle])
fichier.close()
fichier=open("data.txt", "w")    
fichier.write(contenu)
fichier.close()
print(contenu)    
  