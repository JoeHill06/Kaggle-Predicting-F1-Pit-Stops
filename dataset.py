import csv

class dataset():
    def __init__(self, training_path="Data/playground-series-s6e5/train.csv",
                 test_path="Data/playground-series-s6e5/test.csv"):
        self.x_all = []
        self.y_all = []

        self.x_train = []
        self.y_train = []

        self.x_val = []
        self.y_val = []

        self.x_test = []
        self.test_ids = []

        # Label encoders for categorical columns
        self.driver_enc = {}
        self.compound_enc = {}
        self.race_enc = {}

        # columns: id, Driver, Compound, Race, Year, PitStop, LapNumber, Stint,
        #          TyreLife, Position, LapTime (s), LapTime_Delta,
        #          Cumulative_Degradation, RaceProgress, Position_Change, PitNextLap

        with open(training_path, "r", newline="", encoding="utf-8") as f:
            raw_rows = list(csv.reader(f))

        # Build encodings from training data before parsing
        for row in raw_rows[1:]:
            if row[1] not in self.driver_enc:
                self.driver_enc[row[1]] = len(self.driver_enc)
            if row[2] not in self.compound_enc:
                self.compound_enc[row[2]] = len(self.compound_enc)
            if row[3] not in self.race_enc:
                self.race_enc[row[3]] = len(self.race_enc)

        for row in raw_rows[1:]:
            self.x_all.append(self.create_training_row(row))
            self.y_all.append(float(row[-1]))

        with open(test_path, "r", newline="", encoding="utf-8") as f:
            test_rows = list(csv.reader(f))

        for row in test_rows[1:]:
            self.test_ids.append(row[0])
            self.x_test.append(self.create_test_row(row))

    def _enc(self, enc_dict, val):
        return enc_dict.get(val, -1)

    def create_training_row(self, row):
        return [
            self._enc(self.driver_enc, row[1]),   # Driver
            self._enc(self.compound_enc, row[2]), # Compound
            self._enc(self.race_enc, row[3]),      # Race
            float(row[4]),   # Year
            float(row[5]),   # PitStop
            float(row[6]),   # LapNumber
            float(row[7]),   # Stint
            float(row[8]),   # TyreLife
            float(row[9]),   # Position
            float(row[10]),  # LapTime (s)
            float(row[11]),  # LapTime_Delta
            float(row[12]),  # Cumulative_Degradation
            float(row[13]),  # RaceProgress
            float(row[14]),  # Position_Change
        ]

    def create_test_row(self, row):
        return self.create_training_row(row)  # identical structure, no target column
