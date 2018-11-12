import os
import time
import json
from threading import Timer
from pynvml import *
import matplotlib.pyplot as plt

def getGPUMetrics():
    # Metrics accuracy within 5%, see docs:
    # docs.nvidia.com/deploy/nvml-api
    try:
        nvmlInit()
    except err:
        print("Failed to initialize NVML: ", err)
        os.exit(1)
    
    deviceCount = nvmlDeviceGetCount()
    GPUs = [nvmlDeviceGetHandleByIndex(i) for i in range(deviceCount)]
    
    temperatures = []
    fanSpeed = []
    power = []
    memory = []
    
    for i in range(deviceCount):
        temperatures.append(nvmlDeviceGetTemperature(GPUs[i], NVML_TEMPERATURE_GPU))
        memory.append(nvmlDeviceGetMemoryInfo(GPUs[i]).used)
        fanSpeed.append(nvmlDeviceGetFanSpeed(GPUs[i]))
        power.append(nvmlDeviceGetPowerUsage(GPUs[i]) / 1000) # Miliwatt to watt
        
    try:
        nvmlShutdown()
    except err:
        print("Error shutting down NVML:", err) 
        os.exit(1)
        
    metrics = { 'gpu%i'%i:  {
                   'temperatures': temperatures[i],
                   'fanSpeed': fanSpeed[i],
                   'memory': memory[i],
                   'power': power[i],
                   'time': time.strftime("%H:%M:%S")
        } for i in range(deviceCount)
    }   
        
    return metrics
        
        
def saveGPUMetrics(metrics, saving_dir='', name='monitoring_metrics.json', deviceCount=2):
    # Save to json GPU metrics
  
    json_path = os.path.join(saving_dir, name)
    if os.path.exists(json_path):
        # Load existing metrics and add news
        with open(json_path, 'r') as f: 
            old_metrics = json.load(f)
            for i in range(deviceCount):
                for key in old_metrics['gpu%i'%i]:
                    old_metrics['gpu%i'%i][key].append(metrics['gpu%i'%i][key])
        with open(json_path, 'w') as f:            
            json.dump(old_metrics, f, indent=4)  
    else:
        # If does not exist
        with open(json_path, 'w') as f:
            for i in range(deviceCount):
                for key in metrics['gpu%i'%i]:
                    metrics['gpu%i'%i][key] =  [metrics['gpu%i'%i][key]]
            json.dump(metrics, f, indent=4)
    
def plotMetrics(json_path, key_name, limit=93):
    
    with open(json_path, 'r') as f: 
            metrics = json.load(f)
            
            plt.plot(metrics['gpu0'][key_name], label='GPU 0')
            plt.plot(metrics['gpu1'][key_name], label='GPU 1')
            plt.hlines(limit,0, len(metrics['gpu0'][key_name]), color='red', linestyle='--')
            plt.grid()
            plt.legend()
            plt.title(key_name)
            plt.show()

# How to use:
# monitor = monitoring.monitoringGPU(30) # autostart, time in seconds
# - Do fancy stuffs
# monitor.stop()            
class monitoringGPU(object):
    def __init__(self, interval, *args, **kwargs):
        self._timer     = None
        self.interval   = interval
        self.args       = args
        self.kwargs     = kwargs
        self.is_running = False
        self.start()

    def _run(self):
        self.is_running = False
        self.start()
        metrics = getGPUMetrics()
        saveGPUMetrics(metrics, *self.args, **self.kwargs)

    def start(self):
        if not self.is_running:
            self._timer = Timer(self.interval, self._run)
            self._timer.start()
            self.is_running = True

    def stop(self):
        self._timer.cancel()
        self.is_running = False            
          