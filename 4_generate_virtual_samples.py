import itertools
import pandas as pd


if __name__ == '__main__':
    # generate virtual space
    search_range = {"Al": [i / 100 for i in range(0, 11)],
                    "Ti": [i / 100 for i in range(0, 11)],
                    "Fe": [i / 100 for i in range(0, 101, 5)],
                    "Co": [i / 100 for i in range(0, 101, 5)],
                    "Cr": [i / 100 for i in range(0, 101, 5)],
                    "Ni": [i / 100 for i in range(0, 101, 5)],
                    }
    uniques = [i for i in search_range.values()]
    rows = []
    for combination in itertools.product(*uniques):
        if combination[0]+combination[1] <= 0.1 and sum(combination) == 1 :
            if 0 not in combination:
                print(combination)
                rows.append(combination)
    result = pd.DataFrame(rows, columns=search_range.keys())
    result.to_csv("./data/HEA_virtual_samples_simple.csv")