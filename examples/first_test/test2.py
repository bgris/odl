#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 09:43:21 2017

@author: bgris
"""
i=2

if not 2==i:
    
    print('vrai')
    
else: 
    
    print('faux')
    
#%%
a=1
if a<2:
    print("a n'est pas dans l'intervalle.")
else:
    print("a est dans l'intervalle.")    

#%%

def table(nb):

    i = 0

    while i < 10: # Tant que i est strictement inférieure à 10,

        print(i + 1, "*", nb, "=", (i + 1) * nb)

        i += 1 # On incrémente i de 1 à chaque tour de boucle.
        
    #%% 
def table(nb, max):

    """Fonction affichant la table de multiplication par nb
    
    de 1*nb à max*nb
    
    
    
    (max >= 0)"""
        
    i = 0
    
    while i < max:
    
        print(i + 1, "*", nb, "=", (i + 1) * nb)
    
        i += 1

#%%
import math
from math import *
sqrt(25)

#%%

class Compteur:
    """Cette classe possède un attribut de classe qui s'incrémente à chaque
    fois que l'on crée un objet de ce type"""

    
    objets_crees = 0 # Le compteur vaut 0 au départ
    def __init__(self):
        """À chaque fois qu'on crée un objet, on incrémente le compteur"""
        Compteur.objets_crees += 1
    def combien(cls):
        """Méthode de classe affichant combien d'objets ont été créés"""
        print("Jusqu'à présent, {} objets ont été créés.".format(
                cls.objets_crees))
    combien = classmethod(combien)
    
#%%
import Compteur
from Compteur import*
Compteur=Compteur()

#%%
class TableauNoir:
    """Classe définissant une surface sur laquelle on peut écrire,
    que l'on peut lire et effacer, par jeu de méthodes. L'attribut modifié
    est 'surface'"""

    
    def __init__(self):
        """Par défaut, notre surface est vide"""
        self.surface = ""
    def ecrire(self, message_a_ecrire):
        """Méthode permettant d'écrire sur la surface du tableau.
        Si la surface n'est pas vide, on saute une ligne avant de rajouter
        le message à écrire"""

        
        if self.surface != "":
            self.surface += "\n"
        self.surface += message_a_ecrire
    def lire(self):
        """Cette méthode se charge d'afficher, grâce à print,
        la surface du tableau"""

        
        print(self.surface)
    def effacer(self):
        """Cette méthode permet d'effacer la surface du tableau"""
        self.surface = ""
        
#%%

Tableau=TableauNoir()

Tableau1=TableauNoir()




































