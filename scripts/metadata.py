import pandas as pd 
import os 

def CreateLoad(filename='metadata.csv'):
    """
    Creates the Metadata csv, or loads it if it already exists.
    """
    # Retrieve current working directory
    cwd = os.getcwd()
    # Define the path for the metadata file 
    csv_file = os.path.join(cwd, '..', 'data', filename)

    # Check if the metadata file already exists

    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # Create new DataFrame with initial columns
        df = pd.DataFrame(columns=['filename'])

    return df 


def AddRow(df, **kwargs):
    """
    Adds a new row to the DataFrame with the provided keyword arguments.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to which the row will be added.
    **kwargs: dict
        Key-value pairs representing the column names and their corresponding values.
    """
    # Append the new row to the DataFrame
    df = df.append(kwargs, ignore_index=True)
    return df

def AddColumn(df, column_name, default_value=None):
    """
    Adds a new column to the DataFrame with a default value.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to which the column will be added.
    column_name: str
        The name of the new column.
    default_value: any, optional
        The default value for the new column. Default is None.
    
    Returns:
    --------
    pd.DataFrame
        The updated DataFrame with the new column added.
    """
    df[column_name] = default_value
    return df

def SaveMetadata(df, filename='metadata.csv'):
    """
    Saves the DataFrame to a CSV file.
    
    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to be saved.
    filename: str, optional
        The name of the file to save the DataFrame. Default is 'metadata.csv'.
    """
    # Retrieve current working directory
    cwd = os.getcwd()
    # Define the path for the metadata file 
    metadata_path = os.path.join(cwd, '..', 'data', filename)

    # Save the DataFrame to a CSV file
    df.to_csv(metadata_path, index=False)

def DeleteMetadata(filename='metadata.csv', ask:bool=True):
    """
    Deletes the `filename` file if it exists.
    """

    if ask and input("Are you sure you want to delete the "+filename+" file? (y/n): ").lower() != 'y':
        print("Deletion cancelled.")
        return

    # Retrieve current working directory
    cwd = os.getcwd()
    # Define the path for the metadata file 
    csv_file = os.path.join(cwd, '..', 'data', filename)

    # Check if the metadata file exists and delete it
    if os.path.exists(csv_file):
        os.remove(csv_file)
        print(f"Deleted {csv_file}")
    else:
        print(f"{csv_file} does not exist.")
