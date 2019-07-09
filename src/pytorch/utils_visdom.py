from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import plot_learningcurve

# Start web server with: python -m visdom.server

class VisdomWebServer(object):

    def __init__(self):

       DEFAULT_PORT = 8097
       DEFAULT_HOSTNAME = "http://localhost"

       self.vis = Visdom(port=DEFAULT_PORT, server=DEFAULT_HOSTNAME)

    def update(self, metrics):

       if not self.vis.check_connection():
           'No connection could be formed quickly'
           return

       # Learning curve
       try:
              fig, ax = plt.subplots()
              plt.plot(metrics['train_loss'], label='Training loss', color='#32526e')
              plt.plot(metrics['val_loss'], label='Validation loss', color='#ff6b57')
              plt.legend()
              ax.spines['right'].set_visible(False)
              ax.spines['top'].set_visible(False)
              plt.grid(zorder=0, color='lightgray', linestyle='--')
              self.vis.matplot(plt, win='lrcurve')
              plt.close()
              plt.clf()
                
              fig, ax = plt.subplots()
              plt.plot(metrics['learning_rate'], color='#32526e')
              ax.spines['right'].set_visible(False)
              ax.spines['top'].set_visible(False)
              plt.grid(zorder=0, color='lightgray', linestyle='--')
              self.vis.matplot(plt, win='lr_rate')
              plt.close()
              plt.clf()
                
              #plt.figure()
              #plt.plot(metrics['zernike_train_loss'], label='Zernike train loss', color='blue')
              #plt.plot(metrics['zernike_val_loss'], label='Zernike val loss', color='red')
              #plt.legend()
              #plt.grid()
              #self.vis.matplot(plt, win='lrcurve_z')
              #plt.close()
              #plt.clf()  
       except BaseException as err:
              print('Skipped matplotlib example')
              print('Error message: ', err)        
                


if __name__ == "__main__":

       from utils import get_metrics

       metrics = get_metrics('experiments/example')

       visdom = VisdomWebServer()
       visdom.update(metrics)

