import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import math
from scipy.stats import ttest_ind
from sklearn.linear_model import LinearRegression

def df_info(dataframe):
    """
        Function to obtain information and statistics from a DataFrame.

        Args:
            dataframe (pd.Dataframe): Dataframe which we want to obtain the info/statistics.
    """
    print("================================")
    # Obtain the size of the dataframe.
    print("Shape")
    print(dataframe.shape)
    print("--------------------------------")
    # Obtain the number of null values in the dataframe.
    print("Nulls")
    print(dataframe.isnull().sum())
    print("--------------------------------")
    # Obtain concise summary of the dataframe.
    print("Info")
    print(dataframe.info())
    print("--------------------------------")
    # Obtain descriptive statistics of the dataframe.
    print("Descripción")
    print(dataframe.describe())
    print("--------------------------------")
    # Obtain the fist and last 5 rows of the dataframe.
    print("Muestra de registros")
    print(dataframe.head())
    print(dataframe.tail())
    print("================================")

def tests(before_comp_dataframe, after_comp_dataframe):
    """
        Perform a two-sample t test and display the t statistic and p value.

        This function takes two DataFrames, `before_comp_dataframe` and `after_comp_dataframe`, which represent two data samples.
        It then performs a two-sample t-test to compare these two samples and displays the t-statistic and p-value.

        Args:
            before_comp_dataframe (pd.DataFrame): The DataFrame that represents the data sample before a certain event or competition.
            after_comp_dataframe (pd.DataFrame): The DataFrame that represents the data sample after a certain event or competition.

        Note:
            - The two-sample t test is used to determine if there are significant differences between two samples of data.
    """
    # Perform the t test.
    t_stat, p_val = ttest_ind(before_comp_dataframe['demanda'], after_comp_dataframe['demanda'])

    # Print the t statistic and the p value.
    print('Estadístico t:', t_stat)
    print('Valor p:', p_val)

def competition_impact(dataframe):
    """
        Analyze the impact of a competition or event on the data and display statistics before and after the event.

        This function takes a DataFrame containing data over time and performs an analysis to understand the impact of a competition
        or event in the data. Splits the data into two subsets: before the event and after the event, and displays statistics and graphs
        of seasonality for each subset.

        Args:
            dataframe (pd.DataFrame): The DataFrame that contains the data over time.
    """
    # Create a subset with the data before the competitor opens (until 2021-07-02).
    pre_competitor_opening_data = dataframe[dataframe["date"] < '2021-07-02']

    # Create a subset with the data after the competitor opens (from 2021-07-02).
    post_competitor_opening_data = dataframe[dataframe["date"] >= '2021-07-02']

    print("Pre apertura del competidor:")
    df_info(pre_competitor_opening_data)
    montly_seasonality(pre_competitor_opening_data, 'Estacionalidad antes de la apertura')

    print("Post apertura del competidor:")
    df_info(post_competitor_opening_data)
    montly_seasonality(post_competitor_opening_data, 'Estacionalidad después de la apertura')

    # Create the data subsets with the quarters between each year of the data set.
    range_2020 = dataframe[(dataframe["date"] >= '2020-07-01') & (dataframe["date"] <= '2020-09-30')]
    range_2021 = dataframe[(dataframe["date"] >= '2021-07-01') & (dataframe["date"] <= '2021-09-30')]
    range_2022 = dataframe[(dataframe["date"] >= '2022-07-01') & (dataframe["date"] <= '2022-09-30')]

    print("Fecha entre 2020-07-01 y 2020-09-31")
    df_info(range_2020)
    montly_seasonality(range_2020, 'Estacionalidad del trimestre entre entre 2020-07-01 y 2020-09-31')

    print("Fecha entre 2021-07-01 y 2021-09-31")
    df_info(range_2021)
    montly_seasonality(range_2021, 'Estacionalidad del trimestre entre entre 2021-07-01 y 2021-09-31')

    print("Fecha entre 2022-07-01 y 2022-09-31")
    df_info(range_2022)
    montly_seasonality(range_2022, 'Estacionalidad del trimestre entre entre 2022-07-01 y 2022-09-31')

    tests(pre_competitor_opening_data, post_competitor_opening_data)


def histograms(dataframe, columns):
    """
        Function to obtain the histogram from a DataFrame.

        Args:
            dataframe (pf.Dataframe): Dataframe which we want to obtain the histogram.
            column (string): Target column which we want the histogram.
    """

    # Calculate the number of rows and columns in the grid (window)
    if len(columns) <= 2:
        num_rows, num_cols = 1, len(columns)
    else:
        # Obtaining rows considering the number of columns divided by 2, 
        # the number is obtained by rounding up a decimal number to the nearest integer with math.ceil().
        num_rows = math.ceil(len(columns) / 2) 
        num_cols = 2
    
    # Create a figure with subwindows.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    
    # Iterate through the columns and create the histograms.
    for i, column in enumerate(columns):
        # All values found in the column are considered, up to NaN.
        frequency = dataframe[column].value_counts(dropna=False)
        row_index = i // 2  # Row index
        col_index = i % 2   # Column index
        ax = axes[row_index][col_index]
        frequency.plot(kind='bar', ax=ax)
        ax.set_title(f'Histograma de frecuencia de {column}')
        ax.set_xlabel(column)
        ax.set_ylabel('Frecuencia')
    
    # Remove empty subwindows if num_columns is not even.
    if len(columns) % 2 != 0:
        fig.delaxes(axes[-1][-1])
    
    plt.tight_layout()
    # Show the figure
    plt.show()

def correlation_matrix(dataframe):
    """
        Function to obtain the correlation matrix from a DataFrame.

        Args:
            dataframe (pf.Dataframe): Dataframe which we want to obtain the correlation matrix.

        Returns:
            dataframe: Dataframe with the values of the correlation matrix.
    """   

    # Create a heatmap with Seaborn.
    plt.figure(figsize=(12, 8))
    sns.set(font_scale=1.2)
    
    # Create a heatmap with Seaborn.
    sns.heatmap(dataframe.corr(), annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})

    plt.xlabel("Variables")
    plt.ylabel("Variables")
    plt.title('Matriz de Correlación')
    
    plt.show()

    return dataframe.corr()

def montly_seasonality(dataframe, title):
    """
        Function to obtain the montly seasonality from a DataFrame.

        Args:
            dataframe (pf.Dataframe): Dataframe which we want to obtain the seasonality.
    """    
    # Set 'date' as the index.
    dataframe.set_index('date', inplace=True)

    # Group by month and calculate the sum of sales for each month.
    monthly_sales = dataframe.resample('MS').sum()

    # Plot monthly sales over time.
    plt.figure(figsize=(14,10))
    plt.plot(monthly_sales.index, monthly_sales['demanda'])
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.title(title)
    plt.show()

def montly_seasonality_test(dataframe):  
    """
            Function to obtain the montly seasonality from a DataFrame.

            Args:
                dataframe (pf.Dataframe): Dataframe which we want to obtain the seasonality.
    """    
    # Convert date column to date format.
    dataframe["date"] = pd.to_datetime(dataframe["date"]) 
    # Group the data by week and add the values.
    df_semanal = dataframe.resample("W", on="date").sum() 

    columnas = ["demanda", "Regresión lineal", "Ridge lineal", "Lasso lineal", "Random Forest", "XGBoost"]
    df_grafica = df_semanal.loc[:, columnas]
    
    df_grafica.plot(style=["b-", "r--", "g-.", "o-", "p-"])
    plt.legend(loc="upper left")
    plt.xlabel("Semana")
    plt.ylabel("Demanda y predicciones")
    plt.show()
    


def seasonality_per_product(dataframe):
    """
        Analyze the seasonality of monthly sales for different products.

        Args:
            dataframe (pd.DataFrame): The DataFrame that contains the sales data over time.
    """

    # Set 'date' as the index.
    dataframe.set_index('date', inplace=True)

    # Get the list of unique products.
    products = sorted(dataframe['id_producto'].unique())

    # For each product, plot monthly sales over time.
    for product in range(1,7):
        product_sales = dataframe[dataframe['id_producto'] == product]
        product_subcategory = dataframe[dataframe['id_producto'] == product]['subcategoria'].iloc[0]

        monthly_sales = product_sales.resample('M').sum()
        
        plt.figure(figsize=(10,6))
        plt.plot(monthly_sales.index, monthly_sales['demanda'])
        plt.xlabel('Fecha')
        plt.ylabel('Ventas')
        plt.title(f'Ventas mensuales a lo largo del tiempo para el producto con id {product} la cual es de subcategoria {product_subcategory}')
        plt.show()

def tendency(dataframe):
    """
        Analyze the trend of monthly sales over time.

        Args:
            dataframe (pd.DataFrame): The DataFrame that contains the sales data over time.

        Notes:
            - The function uses the 'date' column as a reference to group the data by month.
            - Linear regression is used to estimate the sales trend.
            - Chart shows actual monthly sales and estimated trend line.
     """
    # Set 'date' as the index.
    dataframe.set_index('date', inplace=True)

    # Group by month and calculate the sum of sales for each month.
    monthly_sales = dataframe.resample('M').sum()

    # Create a time matrix for trend line fitting.
    time_matrix = np.arange(len(monthly_sales)).reshape(-1, 1)

    # Initialization of the linear regression model.
    model = LinearRegression()
    model.fit(time_matrix, monthly_sales['demanda'])

    # Predict sales with trend line.
    trend_line = model.predict(time_matrix)

    # Plot sales and trend line.
    plt.figure(figsize=(10,6))
    plt.plot(monthly_sales.index, monthly_sales['demanda'], label='Ventas')
    plt.plot(monthly_sales.index, trend_line, label='Tendencia', linestyle='--')
    plt.xlabel('Fecha')
    plt.ylabel('Ventas')
    plt.title('Ventas mensuales y tendencia a lo largo del tiempo')
    plt.legend()
    plt.show()