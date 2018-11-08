from load_csv import DataSet
from plotter import Plotter


if __name__ == '__main__':
    print("start")
    data = DataSet()

    features = data.get_trainX_pd()
    labels = data.get_trainY_pd()

    plotter = Plotter()

    for i in features.columns :
        plotter.visualizeBarChart(i, "X label", "Y label")
        plotter.visualizeScatterPlot(i, "X label", "Y label", labels)
        plotter.visualizeHistogram(i, "X Label", "Y Label")
        # plotter.visualizeLine(range(70000), i, "X Label", "Y Label")

