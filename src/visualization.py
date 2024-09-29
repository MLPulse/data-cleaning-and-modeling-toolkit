import matplotlib.pyplot as plt
import seaborn as sns

def visualize_missing_data(df):
    """Visualize missing data in the DataFrame."""
    missing_data = df.isnull().sum()

    plt.figure(figsize=(10, 6))
    missing_data.plot(kind='bar')
    plt.title('Missing Data in Each Column')
    plt.xlabel('Columns')
    plt.ylabel('Number of Missing Values')
    plt.show()

    plt.figure(figsize=(14, 10))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Heatmap of Missing Data')
    plt.show()
