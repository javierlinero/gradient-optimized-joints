import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import os
import argparse
from scipy.interpolate import interp1d

# process each csv file outputted from instron
def process_csv(file_path):
    # pandas and data manipulation to put interval between 30N and 60N
    data = pd.read_csv(file_path, skiprows=1)
    data.columns = ['Time', 'Displacement', 'Force', 'Tensile Displacement', 'Strain', 'Stress']
    data['Time'] = pd.to_numeric(data['Time'], errors='coerce')
    data['Displacement'] = pd.to_numeric(data['Displacement'], errors='coerce')
    data['Force'] = pd.to_numeric(data['Force'], errors='coerce')
    data = data[(data['Force'] >= 30) & (data['Force'] <= 60) & (data['Displacement'] != 0)].copy()
    data['Stiffness (N/mm)'] = data['Force'] / data['Displacement']
    
    max_stiffness = data['Stiffness (N/mm)'].max()

    force_stiffness_interp = interp1d(data['Force'], data['Stiffness (N/mm)'], kind='linear', fill_value="extrapolate")
    stiffness_at_30N = force_stiffness_interp(30)
    stiffness_at_60N = force_stiffness_interp(60)

    return data, max_stiffness, stiffness_at_30N, stiffness_at_60N

def calculate_youngs_modulus(data):
    from scipy.stats import linregress
    
    # linear regression model
    slope, intercept, r_value, p_value, std_err = linregress(data['Strain'], data['Stress'])
    
    # slope of the stress-strain curve is the Young's Modulus
    youngs_modulus = slope
    
    return youngs_modulus, r_value**2  


def plot_stiffness(data_dict, joint_name):
    plt.figure(figsize=(10, 8))
    
    # picked out hex colors for blue & orange
    initial_colors = ['#3d5a80', '#98c1d9', '#e0fbfc', '#293241']
    optimized_colors = ['#ffc100', '#ff9a00', '#ff7400', '#ff4d00']

    initial_data = []
    optimized_data = []
    
    for label, data in data_dict.items():
        if 'Initial' in label:
            initial_data.append(data)
        else:
            optimized_data.append(data)
    
    # initial w/ blue
    for i, data in enumerate(initial_data):
        color = initial_colors[i % len(initial_colors)]  # Cycle through colors if more than four datasets
        plt.plot(data['Time'], data['Stiffness (N/mm)'], label=f'{joint_name} Initial {i+1}', color=color)
    
    # optimized w/ orange
    for i, data in enumerate(optimized_data):
        color = optimized_colors[i % len(optimized_colors)]  # Cycle through colors if more than four datasets
        plt.plot(data['Time'], data['Stiffness (N/mm)'], label=f'{joint_name} Optimized {i+1}', color=color)
    
    if initial_data:
        avg_initial = pd.concat(initial_data).groupby('Time').mean()['Stiffness (N/mm)']
        plt.plot(avg_initial.index, avg_initial, label='Average Initial', linewidth=2, color='navy', linestyle='--')
    
    if optimized_data:
        avg_optimized = pd.concat(optimized_data).groupby('Time').mean()['Stiffness (N/mm)']
        plt.plot(avg_optimized.index, avg_optimized, label='Average Optimized', linewidth=2, color='darkorange', linestyle='--')
    
    plt.title(f'Stiffness (N/mm) Over Time for {joint_name.capitalize()}')
    plt.xlabel('Time (s)')
    plt.ylabel('Stiffness (N/mm)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.style.use('ggplot')
    plt.show()

def plot_force_displacement(data_dict, joint_name):
    plt.figure(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, (label, data) in enumerate(data_dict.items()):
        # Filtering data around 30N force
        data_near_30N = data[np.isclose(data['Force'], 30, atol=0.5)]
        
        if not data_near_30N.empty:
            plt.scatter(data_near_30N['Displacement'], data_near_30N['Force'], color=colors[i % len(colors)], label=label)
    
    plt.title(f'Force vs. Displacement at 30N for {joint_name}')
    plt.xlabel('Displacement (mm)')
    plt.ylabel('Force (N)')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def main(opt):
    base_dir = '../instron_data'
    joint_dirs = [d for d in os.listdir(base_dir) if d.endswith('.is_tens_Exports')]

    for joint_dir in joint_dirs:
        joint_name = joint_dir.replace('.is_tens_Exports', '')
        data_dict = {}
        youngs_moduli = {}
        youngs_moduli_initial = []
        youngs_moduli_optimized = []
        max_stiffness_initial = []
        max_stiffness_optimized = []
        stiffness_30N_initial = {}
        stiffness_60N_initial = {}
        stiffness_30N_optimized = {}
        stiffness_60N_optimized = {}
        full_path = os.path.join(base_dir, joint_dir)
        
        for file_name in os.listdir(full_path):
            if file_name.endswith('.csv'):
                iteration_type = 'Optimized' if int(file_name.split('_')[-1].replace('.csv', '')) > 3 else 'Initial'
                label = f'{joint_name} {iteration_type} {file_name.split("_")[-1].replace(".csv", "")}'
                file_path = os.path.join(full_path, file_name)

                data, max_stiff_value, stiff_30N, stiff_60N = process_csv(file_path)
                data_dict[label] = data
                
                # calc young modulus
                youngs_modulus, r_squared = calculate_youngs_modulus(data)
                youngs_moduli[label] = (youngs_modulus, r_squared)

                if iteration_type == 'Initial':
                    max_stiffness_initial.append(max_stiff_value)
                    youngs_moduli_initial.append(youngs_modulus)
                    stiffness_30N_initial[label] = stiff_30N
                    stiffness_60N_initial[label] = stiff_60N
                else:
                    max_stiffness_optimized.append(max_stiff_value)
                    youngs_moduli_optimized.append(youngs_modulus)
                    stiffness_30N_optimized[label] = stiff_30N
                    stiffness_60N_optimized[label] = stiff_60N

        plot_stiffness(data_dict, joint_name)
        #plot_force_displacement(data_dict, joint_name)

        # young's modulus
        if opt.incl_youngs:
            print("-------------------------------------------------------\n")
            print("--= Young's Modulus =-- \n")
            for label, (modulus, r_squared) in youngs_moduli.items():
                print(f"Young's Modulus for {label}: {modulus:.3f} MPa (R^2: {r_squared:.3f})")
            print("")

            if youngs_moduli_initial:
                avg_youngs_initial = np.mean(youngs_moduli_initial)
                print(f"Average Young's Modulus for Initial tests in {joint_name}: {avg_youngs_initial:.3f} MPa")
            
            if youngs_moduli_optimized:
                avg_youngs_optimized = np.mean(youngs_moduli_optimized)
                print(f"Average Young's Modulus for Optimized tests in {joint_name}: {avg_youngs_optimized:.3f} MPa \n")

            if youngs_moduli_initial and youngs_moduli_optimized:
                improvement = ((avg_youngs_optimized - avg_youngs_initial) / avg_youngs_initial) * 100
                improvement_type = "gain" if improvement > 0 else "loss"
                print(f"Improvement Percentage in Young's Modulus for {joint_name}: {improvement:.3f}% ({improvement_type})\n")

        # printing out maximum stiffness (overview)
        if opt.incl_stiffness:
            print("--= Maximum Stiffness Average =-- \n")

            if max_stiffness_initial:
                avg_max_stiff_initial = np.mean(max_stiffness_initial)
                print(f"Average Maximum Stiffness for Initial tests in {joint_name}: {avg_max_stiff_initial:.3f} N/mm")
            if max_stiffness_optimized:
                avg_max_stiff_optimized = np.mean(max_stiffness_optimized)
                print(f"Average Maximum Stiffness for Optimized tests in {joint_name}: {avg_max_stiff_optimized:.3f} N/mm \n")

            if max_stiffness_initial and max_stiffness_optimized:
                improvement = ((avg_max_stiff_optimized - avg_max_stiff_initial) / avg_max_stiff_initial) * 100
                improvement_type = "gain" if improvement > 0 else "loss"
                print(f"Improvement Percentage in Maximum Stiffness for {joint_name}: {improvement:.3f}% ({improvement_type})\n")

            # evaluating at edge cases
            print("--= Stiffness Differences at 30N and 60N =-- \n")
            if stiffness_30N_initial and stiffness_30N_optimized:
                avg_stiff_30N_initial = np.mean(list(stiffness_30N_initial.values()))
                std_stiff_30N_initial = np.std(list(stiffness_30N_initial.values()))
                avg_stiff_30N_optimized = np.mean(list(stiffness_30N_optimized.values()))
                std_stiff_30N_optimized = np.std(list(stiffness_30N_optimized.values()))
                improvement_30N = ((avg_stiff_30N_optimized - avg_stiff_30N_initial) / avg_stiff_30N_initial) * 100
                for label in sorted(stiffness_30N_initial.keys()):
                    print(f"Stiffness at 30N for {label} (Initial): {stiffness_30N_initial[label]:.3f} N/mm")
                for label in sorted(stiffness_30N_optimized.keys()):
                    print(f"Stiffness at 30N for {label} (Optimized): {stiffness_30N_optimized[label]:.3f} N/mm")
                print("")
                print(f"Average Stiffness at 30N for Initial tests in {joint_name}: {avg_stiff_30N_initial:.3f} N/mm ± {std_stiff_30N_initial:.3f}")
                print(f"Average Stiffness at 30N for Optimized tests in {joint_name}: {avg_stiff_30N_optimized:.3f} N/mm ± {std_stiff_30N_optimized:.3f}\n")

                print(f"Improvement Percentage at 30N for {joint_name}: {improvement_30N:.3f}% \n")

            if stiffness_60N_initial and stiffness_60N_optimized:
                avg_stiff_60N_initial = np.mean(list(stiffness_60N_initial.values()))
                std_stiff_60N_initial = np.std(list(stiffness_60N_initial.values()))
                avg_stiff_60N_optimized = np.mean(list(stiffness_60N_optimized.values()))
                std_stiff_60N_optimized = np.std(list(stiffness_60N_optimized.values()))
                improvement_60N = ((avg_stiff_60N_optimized - avg_stiff_60N_initial) / avg_stiff_60N_initial) * 100
                for label in sorted(stiffness_60N_initial.keys()):
                    print(f"Stiffness at 60N for {label} (Initial): {stiffness_60N_initial[label]:.3f} N/mm")
                for label in sorted(stiffness_60N_optimized.keys()):
                    print(f"Stiffness at 60N for {label} (Optimized): {stiffness_60N_optimized[label]:.3f} N/mm")
                print("")
                print(f"Average Stiffness at 60N for Initial tests in {joint_name}: {avg_stiff_60N_initial:.3f} N/mm ± {std_stiff_60N_initial:.3f}")
                print(f"Average Stiffness at 60N for Optimized tests in {joint_name}: {avg_stiff_60N_optimized:.3f} N/mm ± {std_stiff_60N_optimized:.3f}\n")
                print(f"Improvement Percentage at 60N for {joint_name}: {improvement_60N:.3f}% \n")

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--incl_youngs', type=bool, default=False,
                        help='include youngs modulus calculation')
    parser.add_argument('--incl_stiffness', type=bool, default=True,
                        help='include stiffness improvement calculation')
    opt = parser.parse_args()
    main(opt)