import pandas as pd
import plotly.express as px
import numpy as np

desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', None)


#base_path = 'data/2WeekRosterSolutionOptimal'
base_path = 'data/2WeekRosterSolutionOptimal'

roster_solution1_df = pd.read_parquet(f'{base_path}StartCondition.parquet').\
    assign(lastOneWeekRosterIndex=lambda x: x.rosterIndexWeek2,
           lastTwoWeekRosterIndex=lambda x: x.rosterIndex)

nurse1_df = roster_solution1_df.\
    groupby(['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']).\
    agg(nurseCount=('nRostersInSolution', 'sum')).reset_index().astype({'nurseCount': 'int32'})

roster_solution2_df = pd.read_parquet(f'{base_path}.parquet')
nurse2_df = roster_solution2_df.groupby(['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']).\
    agg(nurseCount=('nRostersInSolution', 'sum')).reset_index().astype({'nurseCount': 'int32'})


renaming = {str(i): str(i+14) for i in range(14)} | \
           {'rosterIndexWeek1': 'rosterIndexWeek3', 'rosterIndexWeek2': 'rosterIndexWeek4', 'nRostersInSolution': 'nurseCount', 'totalCost': 'totalCost2'}
merge_columns = ['nurseLevel', 'nurseHours', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']
roster_solution2_renamed_df = roster_solution2_df.rename(columns=renaming)[list(renaming.values())+merge_columns]
four_week_solution_df = roster_solution1_df.merge(roster_solution2_renamed_df, on=merge_columns, how='inner')\
    [[str(i) for i in range(14*2)] + merge_columns + ['nurseCount', 'totalCost', 'totalCost2']]


total_cost1 = (four_week_solution_df['totalCost'] * four_week_solution_df['nurseCount']).sum()
total_cost2 = (four_week_solution_df['totalCost2'] * four_week_solution_df['nurseCount']).sum()
total_cost = total_cost1 + total_cost2
fig = px.imshow(four_week_solution_df[[str(i) for i in range(14*2)]], color_continuous_scale=["blue", "green", "red", 'yellow'],
                title=f'Total cost: {total_cost:.0f}, Total cost1: {total_cost1:.0f}, Total cost2: {total_cost2:.0f}')
fig.show()

# plot demand
day_columns = [str(i) for i in range(14)]
df = four_week_solution_df[['nurseLevel']+day_columns]
df.groupby(['nurseLevel']).value_counts().reset_index()

for i in range(14):
    df[f'week{i}'] = df[str(i)].value_counts().sort_index()
df[['1']]

dfs = []
for i in range(14):
    col = str(i)
    dfs.append(
        df.groupby('nurseLevel')[[col]].value_counts().sort_index().reset_index().loc[lambda x: x[col] != 3].drop(columns=[col]).rename(columns={'count': col})
    )
pd.concat(dfs, axis=1)

df.groupby('nurseLevel', as_index=True).value_counts()