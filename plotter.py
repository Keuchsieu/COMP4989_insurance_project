from load_csv import DataSet

import numpy as np
import matplotlib.pyplot as plt

"""

Class: Plotter
Description: Contains methods to plot and visualize data features.
Featured charts are: Histogram, Bar Chart, Scatter Plot, Line Plot

Version: 1.0.0

"""


class Plotter:

    def __init__(self):
        self.data = DataSet()

    def visualize_bar_chart(self, feature_name, x_label, y_label, data=None) :
        """

        Create a bar chart of the specified feature.

        :param feature_name: Name of the feature to be visualized
        :param x_label: Label for X axis
        :param y_label: Label for Y axis
        :param data: The dataset, defaults to null so sets to overall dataset
        :return: Bar Chart
        """

        if data:
            feature, counts = np.unique(data.get_trainX_pd().loc[:, feature_name], return_counts=True)
        else:
            feature, counts = np.unique(self.data.get_trainX_pd().loc[:, feature_name], return_counts=True)
        plt.bar(feature, counts)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(feature_name + " Bar Chart")
        plt.show()

    def visualize_scatter_plot(self, feature_name, x_label, y_label, label_name, data=None) :
        """

        Create a scatter plot of the specified feature.

        :param feature_name: Name of the feature to be visualized
        :param x_label: Label for X axis
        :param y_label: Label for Y axis
        :param: label_name: Label name
        :return: Scatter Plot
        """
        if data:
            feature = data.get_trainX_pd().loc[:, feature_name]
        else:
            feature = self.data.get_trainX_pd().loc[:, feature_name]
        plt.scatter(feature, label_name)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(feature_name + " Scatter Plot")
        plt.show()

    def visualize_histogram(self, feature_name, x_label, y_label, data=None, bin_num=10) :
        """

        Create a Histogram of the specified feature.

        :param feature_name: Name of the feature to be visualized
        :param x_label: Label for X axis
        :param y_label: Label for Y axis
        :param data: Dataset
        :param binNum: Number of bins
        :return: Histogram
        """
        if data:
            plt.hist(data.get_trainX_pd().loc[:, feature_name], bins=bin_num)
        else:
            plt.hist(self.data.get_trainX_pd().loc[:, feature_name], bins=bin_num)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(feature_name + "Histogram")
        plt.show()

    def visualize_line(self, values, feature_name, x_label, y_label, data=None):
        """

        Create a line plot

        :param values: The range of the values on the Y axis ( I think )
        :param feature_name: The values to plot as a line
        :param x_label: Label for X axis
        :param y_label: Label for Y axis
        :param data: Dataset , default null so overall dataset is used
        :return: Line Plot
        """
        if data:
            plt.plot(values, data.get_trainX_pd().loc[:, feature_name])
        else:
            plt.plot(values, self.data.get_trainX_pd().loc[:, feature_name])
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title("Line Plot")
        plt.show()


    # def visualizeData(self, type, feature, label, x_label, y_label):
    #
    #     if type == "bar":
    #         return self.visualizeBarChart(feature, x_label, y_label)
    #     elif type == "hist":
    #         return self.visualizeHistogram(feature, x_label, y_label)
    #     return self.visualizeScatterPlot(feature, x_label, y_label, label)
