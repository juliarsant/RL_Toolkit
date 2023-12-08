import numpy as np
import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt

'''
Graphs and saves data from experiments. Can choose the titles for the graph, axes, legends.
Can save data by inserting an experiment code.
'''
class DataClass():
    def __init__(self, data_path:str, graph_path:str, num_plots = 1, plot_std = True,axis=0, hitl = False, hitl_demo_num = 0, legend = [], data = []):
        self.data = data
        self.plot_std = plot_std
        self.num_plots = num_plots
        self.axis = axis
        self.hitl = hitl
        self.legend = legend
        self.hitl_demo_num = hitl_demo_num
        self.data_path = data_path
        self.graph_path = graph_path

        self.color = ['#indianred', 'palegreen', 'skyblue', 'mediumslateblue', 'palevioletred', 'dimgrey']
    
    #Chosen Title of the Graph
    def title(self, title:str):
        self.title = title

    #X-Axis Title
    def xaxisTitle(self, title:str):
        self.xaxis_title = title

    #Y-Axis Title
    def yaxisTitle(self, title:str):
        self.yaxis_title = title

    """
    Plot all data in the path
    """
    def plot(self):
        # csv files in the path
        self.data_folder = glob.glob(self.data_path)
        self.graph_folder = glob.glob(self.graph_path)

        data = np.ndarray(self.data)
        shape = data.shape

        #Shape Doesn't Match
        if shape[3] > 0:
            RuntimeError("Data is not the right shape. Only one set of data should be entered: Ex: List of Reward lists over Epsiodes")


        plot_lines = np.ndarray()
        plot_lines_std = np.ndarray()

        for i in range(shape[0]):
            #standard deviation
            std = np.std(data[i], axis=self.axis)  
            #means
            mean = np.mean(data[i], axis = self.axis)

            plot_lines[i] = mean
            plot_lines_std[i] = std
        

        #x values for fill_between()
        x = [i for i in range(len(plot_lines[0]))]

        #Plotting
        for i in range(len(plot_lines[0])):

            plt.plot(x, plot_lines[i], color = self.color[i])
            plt.fill_between(x, plot_lines[i] - plot_lines_std[i], plot_lines[i] + plot_lines_std[i], color = self.color[i], alpha = 0.3)
        
        plt.title(self.title)
        plt.legend(self.legend)
        plt.show()

    """
    Add a new experiement to plot
    """
    def add_data(self, new_data, name):
        self.data.append(new_data)
        self.legend.append(name)

    """
    Save all data inputted
    """
    def save_data(self, experiment_code):
        #save as csv in specific file
        np.savetxt('./Experiment_Data/data_expcode{}.csv'.format(experiment_code), self.data, delimiter = ", ")
        
