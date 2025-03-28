import csv
import yaml

from tests import ROOT_DIR


def read_test_data(name):
    filename = name + ".csv"
    with open(ROOT_DIR / "test_data" / filename, "r") as file:
        lines = csv.reader(file)
        test_data = []
        current_observation = []
        for line in lines:
            if len(line) > 0:
                string_row = list(line)
                row = list(map(float, string_row))
                current_observation.append(row)
            else:
                test_data.append(current_observation)
                current_observation = []

        return test_data


def load_test_config(name):
    filename = name + ".yaml"
    with open(ROOT_DIR / "test_data" / "levels" / filename, "r") as file:
        return yaml.safe_load(file)
