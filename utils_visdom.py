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
              plt.xlim(0, metrics['n_epoch'])
              self.vis.matplot(plt, win='lrcurve')
              plt.close()
              plt.clf()
       except BaseException as err:
              print('Skipped matplotlib example')
              print('Error message: ', err)

       # Information
       metrics_ =  metrics.copy()
       del metrics_['train_loss']
       del metrics_['val_loss']
       metrics_str = json.dumps(metrics_, separators=('<br>', ':'))
       self.vis.text(metrics_str, win='metrics')



if __name__ == "__main__":

       from utils import get_metrics

       metrics = get_metrics('experiments/example')

       visdom = VisdomWebServer()
       visdom.update(metrics)

       # Todo: integration
