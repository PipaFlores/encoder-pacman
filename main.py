#%%
from sqlalchemy import create_engine
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import credentials as cred
#%%MySQL connection
# Replace the connection string with your details
#engine = create_engine(f'mysql+pymysql://{cred.user}:{cred.password}@{cred.host}:{cred.port}/{cred.dbname}')

# Querying the database
#df = pd.read_sql_query("SELECT * FROM game", engine)

#%%
# Reading csv as a df and displaying the first 5 rows
df = pd.read_csv('gamestate.csv', delimiter=';')

# Select data from game_id n
df_8 = df[df['game_id'] == 8]

print(df.head())

# %% Inspecting data
#df["game_id"].unique().tolist().__len__()

#df.types # Check the data types of the columns
# Pacman_X              float64
# Pacman_Y              float64
# Ghost1_X               object
# Ghost1_Y               object
# Ghost2_X              float64
# Ghost2_Y              float64
# Ghost3_X               object
# Ghost3_Y               object
## Loging errors found in the data
df["Ghost1_Y"].describe() 
# Select rows where there is a value, in any row, that includes "touko" string
df[df.astype(str).apply(lambda x: x.str.contains('touko').any(), axis=1)]


# %% Player position heatmap

plt.hist2d(df['Pacman_X'], df['Pacman_Y'], bins=[25, 28], cmap='hot')
plt.colorbar(label='Frequency')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Player Positions')
plt.show()


# %%
# Ghost positions heatmap
plt.hist2d(df['Ghost2_X'], df['Ghost2_Y'], bins=[25, 28], cmap='hot')
plt.colorbar(label='Frequency')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Heatmap of Ghost 1 Positions')
plt.show()


# %%


sns.pairplot(df[['score', 'time_elapsed', 'lives']])
plt.show()
# %%
