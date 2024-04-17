import pandas as pd
import matplotlib.pyplot as plt


def main():
    # loading csv
    file_path = 'optdovescarf1.csv'  # Replace this with your actual file path
    data = pd.read_csv(file_path)

    # set the correct columns
    if data.iloc[0, 1] == '(mm)':
        data = data.drop(index=0).reset_index(drop=True)

    # convert to numeric for the values in case and switch kN to N
    data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
    data['Displacement'] = pd.to_numeric(data['Displacement'], errors='coerce')
    data['Force'] = pd.to_numeric(data['Force'], errors='coerce') * 1000

    #filter to 30N to 60N
    data = data[(data['Force'] >= 30) & (data['Force'] <= 60)]

    # rmv any data where displacement = 0
    data = data[data['Displacement'] != 0]

    # calculate stiffness
    data['Stiffness (N/mm)'] = data['Force'] / data['Displacement']

    # obtain the max stiffness point in the given range
    max_stiffness = data['Stiffness (N/mm)'].max()
    max_time = data[data['Stiffness (N/mm)'] == max_stiffness]['Time'].iloc[0]

    plt.figure(figsize=(12, 6))
    plt.plot(data['Time'], data['Stiffness (N/mm)'], marker='o', linestyle='-', color='blue', label='Stiffness (N/mm)', zorder=1)
    plt.scatter(max_time, max_stiffness, color='red', s=100, edgecolor='black', label=f'Max Stiffness: {max_stiffness:.2f} N/mm at {max_time} s', zorder=2)

    # setup the rest of the plot
    plt.title('Stiffness (Force/Displacement) over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Stiffness (N/mm)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.style.use('ggplot')
    plt.show()



if __name__ == '__main__': 
    main()