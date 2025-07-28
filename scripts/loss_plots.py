import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd 
import numpy as np

#import pandas to read the csv file into a dataframe
#read_csv()

descloss_frame = pd.read_csv("C:/Users/meyer/Desktop/SUVr_Analysis/plots/descrim_loss.csv")
adloss_frame = pd.read_csv("C:/Users/meyer/Desktop/SUVr_Analysis/plots/adversarial_loss.csv")

descloss_x = descloss_frame['Step']
descloss_y = descloss_frame['Value']

adloss_x = adloss_frame['Step']
adloss_y = adloss_frame['Value']

plt.plot(descloss_x, descloss_y, color = "green", label = "Discriminator Loss")  
plt.plot(adloss_x, adloss_y, color = "blue", label = "Adversarial Loss")  
plt.legend(loc = 'upper left', fontsize = 35, frameon = False)
plt.xlabel("Training Iterations", fontsize = 45)
plt.ylabel("Loss Value", fontsize = 45)
plt.xticks(fontsize=32)
plt.yticks(fontsize=32)
plt.show()