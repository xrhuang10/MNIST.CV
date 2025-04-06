#import tensorflow as tf
import numpy as np
#import pandas as pd
#from matplotlib import pyplot as plt

#MAKE A SHOOTER GAME WHERE USER HAS TO SHOOT ONCOMING TARGETS FROM WEBCAM

import pygame

pygame.init()
screen = pygame.display.set_mode((640, 480))
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    screen.fill((0, 0, 0))
    pygame.draw.circle(screen, (255, 0, 0), (320, 240), 50)
    pygame.display.flip()
pygame.quit()