from visdom import Visdom
import numpy as np
import matplotlib.pyplot as plt
import json
from utils import plot_learningcurve

# Test for real time monitoring
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
              plt.figure()
              plt.plot(metrics['train_loss'], label='Training loss', color='blue')
              plt.plot(metrics['val_loss'], label='Validation loss', color='red')
              plt.legend()
              plt.grid()
              self.vis.matplot(plt, win='lrcurve')
              plt.close()
              plt.clf()
                
              plt.figure()
              plt.plot(metrics['zernike_train_loss'], label='Zernike train loss', color='blue')
              plt.plot(metrics['zernike_val_loss'], label='Zernike val loss', color='red')
              plt.legend()
              plt.grid()
              self.vis.matplot(plt, win='lrcurve_z')
              plt.close()
              plt.clf()  
       except BaseException as err:
              print('Skipped matplotlib example')
              print('Error message: ', err)        
                
       # Information
       metrics_ =  metrics.copy()
       del metrics_['train_loss']
       del metrics_['val_loss']
       del metrics_['zernike_train_loss']
       del metrics_['zernike_val_loss']
       metrics_str = json.dumps(metrics_, separators=('<br>', ':'))
       self.vis.text(metrics_str, win='metrics')
        
       # Logs
       #log_path = './logs.log'
       #self.vis.text('Logs visdom: <br>', win='logs')
       #with open(log_path, 'r') as f:
       #    logs = f.read().splitlines()
       #    for i in range(16,0,-1): 
       #        last_lines = logs[-(i+1)]
       #        self.vis.text(last_lines, win='logs', append=True)

if __name__ == "__main__":

       from utils import get_metrics

       metrics = get_metrics('experiments/example')

       visdom = VisdomWebServer()
       visdom.update(metrics)

       # Todo: integration
