from load_csv import DataSet
from plotter import Plotter


if __name__ == '__main__':
    print("start")
    data = DataSet()

    features = data.get_trainX_pd()
    labels = data.get_trainY_pd()

    plotter = Plotter()

    for i in features.columns :
        plotter.visualize_bar_chart(i, "X label", "Y label")
        plotter.visualize_scatter_plot(i, "X label", "Y label", labels)
        plotter.visualize_histogram(i, "X Label", "Y Label")
        # plotter.visualize_line(range(70000), i, "X Label", "Y Label")

