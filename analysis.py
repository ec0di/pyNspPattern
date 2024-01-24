import pandas as pd
import plotly.express as px
import numpy as np
from helpers import BASE_DEMAND
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"


desired_width = 320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', None)


# plot roster cost examples
roster_df = pd.read_parquet('data/2WeekRosters.parquet')
roster_df.sort_values('totalCost', inplace=True)
roster_df.totalCost.hist(x='totalCost', title='Roster Cost Histogram').show()

# plot worst, best, and top 25 worst roster
df = roster_df.iloc[[0, int(0.25*roster_df.shape[0]), -1]].reset_index()
fig = px.imshow(df[[str(i) for i in range(14)]], color_continuous_scale=["blue", "green", "red", 'yellow']
                , title='Roster Examples')
fig.show()

# Plots Results
solution_base_file = '2WeekRosterSolutionOptimalTest'


def get_solution_df(solution_base_file):

    base_path = f'data/{solution_base_file}'

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
        [[str(i) for i in range(14*2)] + merge_columns + ['nurseCount', 'totalCost', 'totalCost2', 'rosterIndexWeek2', 'rosterIndexWeek3']]
    four_week_solution_df = four_week_solution_df.assign(oneWeekRosterIndices=lambda x: x.rosterIndexWeek2.astype(str) + ', ' + x.rosterIndexWeek3.astype(str),
                                                         rosterCostPerNurse=lambda x: (x.totalCost + x.totalCost2) / x.nurseCount,
                                                         nurseGroup=lambda x: x.nurseHours.astype(str) + ', ' + x.nurseLevel.astype(str)
                                                         )
    four_week_solution_df = four_week_solution_df.sort_values(['rosterCostPerNurse', 'nurseHours', 'nurseLevel']).reset_index(drop=True)

    # create a row for each nurse in nurseCount
    four_week_solution_df = pd.DataFrame(np.repeat(four_week_solution_df.values, four_week_solution_df['nurseCount'], axis=0),
                                         columns=four_week_solution_df.columns)\
                            .astype(dtype=four_week_solution_df.dtypes.astype(str).to_dict())
    four_week_solution_df = four_week_solution_df.assign(nurseCount=1).reset_index(drop=True)
    return four_week_solution_df


four_week_solution_df = get_solution_df(solution_base_file)


def get_solution_schedule_fig(four_week_solution_df, solution_base_file):
    total_cost1 = (four_week_solution_df['totalCost'] * four_week_solution_df['nurseCount']).sum()
    total_cost2 = (four_week_solution_df['totalCost2'] * four_week_solution_df['nurseCount']).sum()
    total_cost = total_cost1 + total_cost2
    fig = px.imshow(four_week_solution_df[[str(i) for i in range(14*2)]], color_continuous_scale=["blue", "green", "red", 'yellow'],
                    title=f'Total Cost: {total_cost:.0f}, Total Cost Period1: {total_cost1:.0f}, Total Cost Period2: {total_cost2:.0f} | Solution File: {solution_base_file}',)
    custom_data = np.stack([np.transpose(np.repeat(np.array([four_week_solution_df['oneWeekRosterIndices']]), 28, axis=0)),
                            np.transpose(np.repeat(np.array([four_week_solution_df['nurseCount']]), 28, axis=0)),
                            np.transpose(np.repeat(np.array([four_week_solution_df['rosterCostPerNurse'].round(2)]), 28, axis=0)),
                            np.transpose(np.repeat(np.array([four_week_solution_df['nurseLevel']]), 28, axis=0)),
                            np.transpose(np.repeat(np.array([four_week_solution_df['nurseHours']]), 28, axis=0))
                            ], axis=-1)
    fig.update(data=[{'customdata': custom_data,
                      'hovertemplate': "rosterCostPerNurse: %{customdata[2]}<br>"
                                       "nurseCount: %{customdata[1]}<br>"
                                       "nurseLevel: %{customdata[3]}<br>"
                                       "nurseHours: %{customdata[4]}<br>"
                                       "day: %{x}<br>"
                                       "roster: %{y}<br>"
                                       "oneWeekRosterIndices: %{customdata[0]}<br>",
                      }])
    colorbar = dict(thickness=25,
                     tickvals=[0, 1, 2, 3],
                     ticktext=['Day', 'Evening', 'Night', 'Off'])
    fig.update(layout_coloraxis_showscale=True, layout_coloraxis_colorbar=colorbar)
    return fig
fig = get_solution_schedule_fig(four_week_solution_df, solution_base_file)
fig.show()

# demand and supply plots
df = four_week_solution_df[[str(i) for i in range(14 * 2)]].transpose().\
    assign(day=lambda x: (x==0).sum(axis=1),
           evening=lambda x: (x==1).sum(axis=1),
           night=lambda x: (x==2).sum(axis=1))
z_all = df[['day', 'evening', 'night']].values.T

# demand plot
average_demand = sum(sum(BASE_DEMAND))
average_supply = z_all.sum() // 4
fig = make_subplots(3, 2, horizontal_spacing=0.15, subplot_titles=['Demand Per Week', '', 'Supply Week 1', 'Supply Week 2', 'Supply Week 3', 'Supply Week 4'])
colorscale = 'Brwnyl'
fig.add_trace(go.Heatmap(z=BASE_DEMAND,
                  colorscale=colorscale,
                  x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                  y=['Day', 'Evening', 'Night'],
                  zmin=z_all.min(),
                  zmax=z_all.max(),
                  colorbar=None), 1, 1)

# supply plots
for i in range(4):
    s1, s2 = str(i*7), str((i+1)*7-1)
    z = df.loc[s1:s2, ['day', 'evening', 'night']].values.T
    if i == 3:
        fig.add_trace(go.Heatmap(z=z,
                                 x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                 y=['Day', 'Evening', 'Night'],
                                 colorscale=colorscale,
                                 colorbar_x=0.45,
                                 zmin=z_all.min(),
                                 zmax=z_all.max(),
                                 ), i//2+2, i%2+1)
    else:
        fig.add_trace(go.Heatmap(z=z,
                                 x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                 y=['Day', 'Evening', 'Night'],
                                 colorscale=colorscale,
                                 zmin=z_all.min(),
                                 zmax=z_all.max(),
                                 colorbar=None), i // 2 + 2, i % 2 + 1)
        fig.update(layout_coloraxis_showscale=False)
fig.update_layout(title_text=f'Nurse Demand and Supply | Solution File: {solution_base_file} | Average Weekly Demand: {average_demand}, Average Weekly Supply: {average_supply}')

fig.show()

# nurse surplus plot
fig = make_subplots(2, 2, horizontal_spacing=0.15, subplot_titles=['Week 1', 'Week 2', 'Week 3', 'Week 4'])
z_all = df[['day', 'evening', 'night']].values.T - np.tile(BASE_DEMAND, (1, 4))
average_surplus = z_all.sum() // 4
colorscale = 'Oryel'
for i in range(4):
    s1, s2 = str(i*7), str((i+1)*7-1)
    z = df.loc[s1:s2, ['day', 'evening', 'night']].values.T - BASE_DEMAND
    if i == 3:
        fig.add_trace(go.Heatmap(z=z,
                                 x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                 y=['Day', 'Evening', 'Night'],
                                 colorscale=colorscale,
                                 colorbar_x=0.45,
                                 zmin=z_all.min(),
                                 zmax=z_all.max(),
                                 ), i//2+1, i%2+1)
    else:
        fig.add_trace(go.Heatmap(z=z,
                                 x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                 y=['Day', 'Evening', 'Night'],
                                 colorscale=colorscale,
                                 zmin=z_all.min(),
                                 zmax=z_all.max(),
                                 colorbar=None), i // 2 + 1, i % 2 + 1)
        fig.update(layout_coloraxis_showscale=False)
fig.update_layout(title_text=f'Nurse Surplus (Supply - Demand) | Solution File: {solution_base_file} |  Average Weekly Surplus: {average_surplus}')
fig.show()


# compare solutions cost plots
df1 = get_solution_df('2WeekRosterSolutionOptimalTest').assign(solution='Optimal')
df2 = get_solution_df('2WeekRosterSolutionTest').assign(solution='Column Generation')

# compare with solutions from column generation
fig = get_solution_schedule_fig(df2, '2WeekRosterSolutionTest')
fig.show()

df = pd.concat([df1, df2])
px.histogram(df, x='rosterCostPerNurse', color='solution', title='Roster Cost Comparison for Optimal and Column Generation Solutions', barmode='group').show()