import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)





#'CMAPSSData/train_FD001.txt' = Traning dataset 1st engine fleet
#'CMAPSSData/test_FD001.txt' = Test dataset 1st engine fleet

def column_converter(dataframe: str):
    """#Function to load dataset convert feature name to their real-world sensor names

    Args:
        dataframe (str): location of the dataset

    Returns:
        _type_: pandas dataframe and list of sensor names
    """    ''''''

    try:

        # Load data
        df = pd.read_csv(dataframe,delim_whitespace=True)
    
    except Exception as e:
        print('f Exception thrown: {e}')
    
    else:
        print('no failures converting dataframe columns')
    

        # List of sensor names
        sensor_names =['Fan inlet temperature ◦R', 'LPC outlet temperature ◦R', 
        'HPC outlet temperature ◦R', 'LPT outlet temperature ◦R',
        'Fan inlet Pressure psia', 'bypass-duct pressure psia',
        'HPC outlet pressure psia', 'Physical fan speed rpm',
        'Physical core speed rpm', 'Engine pressure ratioP50/P2',
        'HPC outlet Static pressure psia', 'Ratio of fuel flow to Ps30 pps/psia',
        'Corrected fan speed rpm', 'Corrected core speed rpm', 'Bypass Ratio ', 
        'Burner fuel-air ratio', 'Bleed Enthalpy', 'Required fan speed', 
        'Required fan conversion speed', 'High-pressure turbines Cool air flow', 
        'Low-pressure turbines Cool air flow']

        #Setting dict to store new names for colummns 
        operational_dict = {
            '1':'Engine',
            '1.1':'Cycles',
            '-0.0007': 'Operational Setting 1',
            '-0.0004': 'Operational Setting 2',
            '100.0': 'Operational Setting 3',
        }

        #Creating a list of columns names
        old_columns = list(df.columns)

        #Removing column names found in operational_dict
        operational_columns_to_rename = [col for col in old_columns if col not in operational_dict.keys()]

        #Combining the two lists into a dict using tuple type casting
        converged_names = dict(map(lambda i,j : (i,j) , operational_columns_to_rename,sensor_names))

        # Combine both dictionaries (operational_dict and converged_names) ** is just short hand combining of the dictionary
        all_rename_dict = {**operational_dict, **converged_names}

        # Renaming columns in the dataframe
        df.rename(columns=all_rename_dict, inplace=True)
        print(df.info())
        print(f'Dataset has been loaded and converted {dataframe}')
        return df, sensor_names


def feature_skew_check(dataframe:str, real_sensor_names: list):

    """takes in dataframe and list of sensornames and plots distributions of each feature and shows data skew.
    """    ''''''

    columns_to_normalize = []

    # Determine the number of rows and columns for the subplot grid
    num_columns = len([col for col in dataframe.columns if col in real_sensor_names])
    num_rows = int(np.ceil(num_columns / 3)) #Making columns

    # Create subplots
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 5))  
    axes = axes.flatten()  # Flatten to easily iterate over it

    # Loop through each column and create an individual plot (only sensor names)
    for i, column in enumerate(dataframe[real_sensor_names]):
        if column in real_sensor_names:
            sns.histplot(data=dataframe[column], ax=axes[i])
            axes[i].set_title(f'Distribution of {column}')
            skewness = dataframe[column].skew()
            print(f'Skewness of {column}: {skewness}')
            
            # Check if the skewness is outside the range of -0.5 to 0.5
            if skewness < -0.5 or skewness > 0.5:
                columns_to_normalize.append(column)

    # Hide any unused subplots if there are any
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    # Print the columns that need normalization
    print(f'Columns that need normalization: {columns_to_normalize}')


test1 = column_converter('CMAPSSData/train_FD001.txt')
test2 = feature_skew_check(test1)







