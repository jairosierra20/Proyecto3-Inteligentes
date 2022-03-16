# ghostAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from itertools import count
import string
from game import Agent
from game import Actions
from game import Directions
import random
from util import manhattanDistance
import util
import tensorflow as tf
import numpy as np


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):

    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 1
        if isScared:
            speed = 0.5

        actionVectors = [Actions.directionToVector(
            a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(
            pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
            legalActions, distancesToPacman) if distance == bestScore]
        #Parametros para entranamiento1
        pacman2 = np.array([distancesToPacman], dtype=float)
        fantasma = np.array([bestProb], dtype=float)
        #Parametros para entrenamiento2
        ghosPos= np.array(pos, dtype=float)
        pacpos= np.array(pacmanPosition, dtype=float)
        #Crear el modelo de prediccion con keras
        capa = tf.keras.layers.Dense(units=1, input_shape=[1])
        modelo = tf.keras.Sequential([capa])
        modelo.compile(
            optimizer=tf.keras.optimizers.Adam(0.1),
            loss='mean_squared_error')

        #Pruebas de prints, para visualizar valores y patrones    
        print("Aqui "+str(distancesToPacman))
        ghost = []
        ghost.append(str(pos)+str(bestActions))
        pacman = []
        pacman.append(str(pacmanPosition)+str(bestActions))
      
        
 
        # Construct distribution
        print("hola2: "+str(bestProb))

        modelo = tf.keras.models.load_model('prueba2.h5')
        print("Hagamos una predicción!")
        bestProb = modelo.predict([bestProb])

        print("hola1: "+str(bestProb))

        print("Hola3: "+str(bestActions))
        print("Hola4: "+str(legalActions))
        print("Hola5: "+str(distancesToPacman))

        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
        dist.normalize()

        return dist
