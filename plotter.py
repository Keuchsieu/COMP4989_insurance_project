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

    def visualizeBarChart(self, featureName, XLabel, YLabel) :
        """

        Create a bar chart of the specified feature.

        :param featureName: Name of the feature to be visualized
        :param XLabel: Label for X axis
        :param YLabel: Label for Y axis
        :return: Bar Chart
        """
        feature, counts = np.unique(self.data.get_trainX_pd().loc[:, featureName], return_counts=True)
        plt.bar(feature, counts)
        plt.xlabel(XLabel)
        plt.ylabel(YLabel)
        plt.title(featureName + " Bar Chart")
        plt.show()

    def visualizeScatterPlot(self, featureName, XLabel, YLabel, labelName) :
        """

        Create a scatter plot of the specified feature.

        :param featureName: Name of the feature to be visualized
        :param XLabel: Label for X axis
        :param YLabel: Label for Y axis
        :return: Scatter Plot
        """

        feature = self.data.get_trainX_pd().loc[:, featureName]
        plt.scatter(feature, labelName)
        plt.xlabel(XLabel)
        plt.ylabel(YLabel)
        plt.title(featureName + " Scatter Plot")
        plt.show()

    def visualizeHistogram(self, featureName, XLabel, YLabel, binNum=10) :
        """

        Create a Histogram of the specified feature.

        :param featureName: Name of the feature to be visualized
        :param XLabel: Label for X axis
        :param YLabel: Label for Y axis
        :param binNum: Number of bins
        :return: Histogram
        """

        plt.hist(self.data.get_trainX_pd().loc[:, featureName], bins=binNum)
        plt.xlabel(XLabel)
        plt.ylabel(YLabel)
        plt.title(featureName)
        plt.show()

    def visualizeLine(self, values, featureName, XLabel, YLabel):
        """

        Create a line plot

        :param values: The range of the values on the Y axis ( I think )
        :param plotValue: The values to plot as a line
        :param XLabel: Label for X axis
        :param YLabel: Label for Y axis
        :return: Line Plot
        """

        plt.plot(values, self.data.get_trainX_pd().loc[:, featureName])
        plt.xlabel(XLabel)
        plt.ylabel(YLabel)
        plt.title("Line Plot")
        plt.show()
