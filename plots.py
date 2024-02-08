import pandas as pd
import plotly.express as px
import numpy as np
from input_parameters import BASE_DEMAND
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# Plots Results
def merge_solution_files(solution_base_file, n_weeks, explode_nurse_count=False):
    merge_columns = ['nurseLevel', 'nurseHours', 'twoWeekRosterIndexHistory']
    renaming = {str(i): str(i) for i in range(14)} | \
               {'totalCost': f'totalCost0',
                'nRostersInSolution': f'nurseCount0'}
    df = pd.read_parquet(f'{solution_base_file}Week0to1.parquet') \
        .assign(twoWeekRosterIndexHistory=lambda x: x[f'twoWeekRosterIndexHistory'] + ', ' + x.rosterIndex.astype(str), ) \
        .rename(columns=renaming)[sorted(list(renaming.values())) + merge_columns]
    for j in range(1, n_weeks // 2):
        df_next = pd.read_parquet(f'{solution_base_file}Week{j*2}to{j*2+1}.parquet')
        renaming = {str(i): str(i + 14*j) for i in range(14)} | \
                   {'nRostersInSolution': f'nurseCount{j}',
                    'totalCost': f'totalCost{j}',}
        df_next = df_next.rename(columns=renaming)[sorted(list(renaming.values()))+merge_columns+['rosterIndex']]
        df = df.merge(df_next, on=merge_columns, how='inner')
        # prepare for next join
        df = df.assign(twoWeekRosterIndexHistory=lambda x:
                    x[f'twoWeekRosterIndexHistory'] + ', ' + x.rosterIndex.astype(str), ).drop(columns='rosterIndex')

    df = df.assign(nurseCount=lambda x: x[f'nurseCount{n_weeks//2-1}'],
                   rosterCostPerNurse=lambda x: x[[f'totalCost{i}' for i in range(n_weeks // 2)]].sum(axis=1) / x.nurseCount,
                   nurseGroup=lambda x: x.nurseHours.astype(str) + ', ' + x.nurseLevel.astype(str))\
        .sort_values(['rosterCostPerNurse', 'nurseHours', 'nurseLevel']).reset_index(drop=True)

    if explode_nurse_count:  # create a row for each nurse in nurseCount
        df = pd.DataFrame(np.repeat(df.values, df.nurseCount, axis=0),
                                             columns=df.columns)\
                                .astype(dtype=df.dtypes.astype(str).to_dict())
        df = df.assign(nurseCount=1).reset_index(drop=True)
    return df


def visualize_optimized_nurse_schedule(solution_base_file, n_weeks, explode_nurse_count):
    df = merge_solution_files(solution_base_file, n_weeks, explode_nurse_count)

    title_str = ''
    total_cost = 0
    for i in range(n_weeks // 2):
        total_cost_i = (df[f"totalCost{i}"] * df['nurseCount']).sum()
        total_cost += total_cost_i
        title_str += f'Total Cost Period{i}: {total_cost_i:.0f}, '
    title_str = f'Total Cost: {total_cost:.0f} | ' + title_str + f' | Solution Base Filename: {solution_base_file}'
    n_days = n_weeks * 7
    fig = px.imshow(df[[str(i) for i in range(n_days)]], color_continuous_scale=["blue", "green", "red", 'yellow'],
                    title=title_str,)
    custom_data = np.stack([np.transpose(np.repeat(np.array([df['twoWeekRosterIndexHistory']]), n_days, axis=0)),
                            np.transpose(np.repeat(np.array([df['nurseCount']]), n_days, axis=0)),
                            np.transpose(np.repeat(np.array([df['rosterCostPerNurse'].round(2)]), n_days, axis=0)),
                            np.transpose(np.repeat(np.array([df['nurseLevel']]), n_days, axis=0)),
                            np.transpose(np.repeat(np.array([df['nurseHours']]), n_days, axis=0))
                            ], axis=-1)
    fig.update(data=[{'customdata': custom_data,
                      'hovertemplate': "rosterCostPerNurse: %{customdata[2]}<br>"
                                       "nurseCount: %{customdata[1]}<br>"
                                       "nurseLevel: %{customdata[3]}<br>"
                                       "nurseHours: %{customdata[4]}<br>"
                                       "day: %{x}<br>"
                                       "roster: %{y}<br>"
                                       "twoWeekRosterIndexHistory: %{customdata[0]}<br>",
                      }])
    colorbar = dict(thickness=25,
                     tickvals=[0, 1, 2, 3],
                     ticktext=['Day', 'Evening', 'Night', 'Off'])
    fig.update(layout_coloraxis_showscale=True, layout_coloraxis_colorbar=colorbar)
    fig.show()
    return fig


def visualize_optimized_nurse_demand_surplus(solution_base_file, n_weeks):

    df_base = merge_solution_files(solution_base_file, n_weeks, explode_nurse_count=True)
    fig = make_subplots(n_weeks // 2, 2,
                        horizontal_spacing=0.15, subplot_titles=[f'Week {i}' for i in range(n_weeks)])
    df = df_base[[str(i) for i in range(n_weeks * 7)]].transpose(). \
        assign(day=lambda x: (x == 0).sum(axis=1),
               evening=lambda x: (x == 1).sum(axis=1),
               night=lambda x: (x == 2).sum(axis=1))
    z_all = df[['day', 'evening', 'night']].values.T - np.tile(BASE_DEMAND, (1, n_weeks))
    average_weekly_surplus = z_all.sum() / n_weeks
    average_daily_surplus = z_all.sum() / n_weeks / 7
    average_daily_per_nurses_surplus = z_all.sum() / n_weeks / 7 / df_base.nurseCount.sum()
    colorscale = 'Oryel'
    for i in range(n_weeks):
        s1, s2 = str(i * 7), str((i + 1) * 7 - 1)
        z = df.loc[s1:s2, ['day', 'evening', 'night']].values.T - BASE_DEMAND
        if i == n_weeks - 1:
            fig.add_trace(go.Heatmap(z=z,
                                     x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                     y=['Day', 'Evening', 'Night'],
                                     colorscale=colorscale,
                                     colorbar_x=0.45,
                                     zmin=z_all.min(),
                                     zmax=z_all.max(),
                                     ), i // 2 + 1, i % 2 + 1)
        else:
            fig.add_trace(go.Heatmap(z=z,
                                     x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                                     y=['Day', 'Evening', 'Night'],
                                     colorscale=colorscale,
                                     zmin=z_all.min(),
                                     zmax=z_all.max(),
                                     colorbar=None), i // 2 + 1, i % 2 + 1)
            fig.update(layout_coloraxis_showscale=False)
    fig.update_layout(
        title_text=f'Nurse Surplus (Supply - Demand) | Solution File: {solution_base_file}'
                   f' | Average Weekly Surplus: {average_weekly_surplus:.1f}'
                   f' | Average Daily Surplus: {average_daily_surplus:.1f}'
                   f' | Average Daily Surplus Per 10 Nurses: {average_daily_per_nurses_surplus:.2f}',)
    fig.show()
    return fig
