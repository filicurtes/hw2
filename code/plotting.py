import matplotlib.pyplot as plt
import os 

def plot(data_to_plot_1,data_to_plot_2,y_name,y_1,y_2,title,file_title,set):
  owd=os.getcwd()
  os.chdir('graphs')
  plt.figure(figsize=(10, 7))
  plt.plot(data_to_plot_1,color='blue', label=y_1)
  plt.plot(data_to_plot_2,color='orange', label=y_2)
  plt.xlabel('Epochs')
  plt.ylabel(y_name)
  plt.legend()
  plt.title(title)
  plt.savefig(f'{set}_{file_title}')
  os.chdir(owd)