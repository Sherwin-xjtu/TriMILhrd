import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_name = 'F:/shenzhen/Sherwin/HRD/all_scaler.csv'
    pd.read_csv(file_name, low_memory=False)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
