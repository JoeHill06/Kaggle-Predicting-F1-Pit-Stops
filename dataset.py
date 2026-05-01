import csv

class dataset():
    def __init__(self, training_path = "Data/playground-series-s6e5/train.csv", test_path = "/Users/joehill/Developer/Kaggle-Predicting-F1-Pit-Stops/Data/playground-series-s6e5/test.csv"):
        # all data
        x_all = []
        y_all = []
 
        # training data and labels
        self.x_train = []
        self.y_train = []

        # validation data and labels
        self.x_val = []
        self.y_val = []

        # test data and results
        self.x_test = []
        self.y_test = []

        with open(training_path, "r", newline= "", encoding="utf-8") as file:
            reader = csv.reader(file)
            raw_rows = list(reader)

        # headers 'id', 'Driver', 'Compound', 'Race', 'Year', 'PitStop', 'LapNumber',
        #  'Stint', 'TyreLife', 'Position', 'LapTime (s)', 'LapTime_Delta',
        #  'Cumulative_Degradation', 'RaceProgress', 'Position_Change', 'PitNextLap'
        for i in range(1, len(raw_rows)):
            raw_row = raw_rows[i]
            raw_result = raw_row[-1]
            self.all_x.append(self.create_training_row(raw_row))
            self.all_y.append(float(raw_result))
            
    
    def create_training_row(self, row):
        pass



data = dataset()