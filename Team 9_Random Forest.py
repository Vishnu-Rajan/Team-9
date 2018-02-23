# Required Python Packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# File Paths
Path = "https://raw.githubusercontent.com/Vishnu-Rajan/Team-9/master/Data.csv"

HEADERS = ["id","ps_ind_01","ps_ind_02_cat","ps_ind_03","ps_ind_04_cat",
          "ps_ind_05_cat","ps_ind_06_bin","ps_ind_07_bin","ps_ind_08_bin",
          "ps_ind_09_bin","ps_ind_10_bin","ps_ind_11_bin","ps_ind_12_bin",
          "ps_ind_13_bin","ps_ind_14","ps_ind_15","ps_ind_16_bin","ps_ind_17_bin",
          "ps_ind_18_bin","ps_reg_01","ps_reg_02","ps_reg_03","ps_car_01_cat","ps_car_02_cat",
          "ps_car_03_cat","ps_car_04_cat","ps_car_05_cat","ps_car_06_cat","ps_car_07_cat",
          "ps_car_08_cat","ps_car_09_cat","ps_car_10_cat","ps_car_11_cat","ps_car_11",
          "ps_car_12","ps_car_13","ps_car_14","ps_car_15","ps_calc_01","ps_calc_02",
          "ps_calc_03","ps_calc_04","ps_calc_05","ps_calc_06","ps_calc_07","ps_calc_08",
          "ps_calc_09","ps_calc_10","ps_calc_11","ps_calc_12","ps_calc_13","ps_calc_14","ps_calc_15_bin",
          "ps_calc_16_bin","ps_calc_17_bin","ps_calc_18_bin","ps_calc_19_bin","ps_calc_20_bin","target"
          ]


def split_dataset(dataset, train_percentage, feature_headers, target_header):
    """
    Split the dataset with train_percentage
    :param dataset:
    :param train_percentage:
    :param feature_headers:
    :param target_header:
    :return: train_x, test_x, train_y, test_y
    """

    train_x, test_x, train_y, test_y = train_test_split(dataset[feature_headers], dataset[target_header],
                                                        train_size=train_percentage, random_state=42)
    return train_x, test_x, train_y, test_y


def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf


def main():
    """
    Main function
    :return:
    """
    # Load the csv file into pandas dataframe
    dataset = pd.read_csv(Path)

    # Split dataset into train and test dataset
    train_x, test_x, train_y, test_y = split_dataset(dataset, 0.80, HEADERS[:58], HEADERS[-1])

    # Train and Test dataset size details
    print "Train_x Shape :: ", train_x.shape
    print "Train_y Shape :: ", train_y.shape
    print "Test_x Shape :: ", test_x.shape
    print "Test_y Shape :: ", test_y.shape

    # Create random forest classifier instance
    trained_model = random_forest_classifier(train_x, train_y)
    print "Trained model :: ", trained_model
    predictions = trained_model.predict(test_x)

    for i in xrange(0, 10):
        print "Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i])

    print "Random Forest Classifier"
    print "Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x))
    print "Test Accuracy  :: ", accuracy_score(test_y, predictions)
    print "Confusion matrix ::\n", confusion_matrix(test_y, predictions)


if __name__ == "__main__":
    main()