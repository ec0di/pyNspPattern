import pandas as pd
import plotly.express as px
import numpy as np
from input_parameters import BASE_DEMAND
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pd.options.plotting.backend = "plotly"

# plot roster cost examples
roster_df = pd.read_parquet('data/2WeekRosters.parquet')
roster_df.sort_values('totalCost', inplace=True)
roster_df.totalCost.hist(x='totalCost', title='Roster Cost Histogram').show()

# plot worst, best, and top 25 worst roster
df = roster_df.iloc[[0, int(0.25*roster_df.shape[0]), -1]].reset_index()
fig = px.imshow(df[[str(i) for i in range(14)]], color_continuous_scale=["blue", "green", "red", 'yellow']
                , title='Roster Examples')
fig.show()

# compare solutions cost plots
df1 = get_solution_df('2WeekRosterSolutionOptimalTest').assign(solution='Optimal')
df2 = get_solution_df('2WeekRosterSolutionTest').assign(solution='Column Generation')

# compare with solutions from column generation
fig = get_solution_schedule_fig(df2, '2WeekRosterSolutionTest')
fig.show()

df = pd.concat([df1, df2])
px.histogram(df, x='rosterCostPerNurse', color='solution', title='Roster Cost Comparison for Optimal and Column Generation Solutions', barmode='group').show()



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





# read in solution
nurse_df_base = pd.read_excel(base_path + 'data/NurseData.xlsx', sheet_name="personindstillinger")
nurse_df = nurse_df_base.groupby(['nurseHours', 'nurseLevel']).agg(nurseCount=('Person', 'count')).reset_index()\
    .rename_axis('nurseType').reset_index()


roster_matching_file = f'data/1WeekRosterMatching.json'
with open(roster_matching_file, 'r') as fp:
    roster_matching = json.load(fp)
    roster_matching = {int(key): value for key, value in roster_matching.items()}

rosters_allowed_after = []
for roster_index in roster_solution1_df.rosterIndexWeek2:
    rosters_allowed_after.append(roster_solution2_df.loc[lambda x: x.rosterIndexWeek1.isin(roster_matching[roster_index]['rostersAllowedAfter'])].rosterIndexWeek1.tolist())

roster_solution1_df['rostersAllowedAfter'] = rosters_allowed_after


roster_solution1_df[['8', '9', '10', '11', '12','13', 'rosterIndexWeek2', 'rostersAllowedAfter', 'nurseLevel', 'nurseHours', 'nRostersInSolution']]

roster_solution2_df[['0', '1', '2', '3', '4', '5', 'rosterIndexWeek1', 'nurseLevel', 'nurseHours']]

renaming = {'0': '14', '1': '15', '2': '16', '3': '17', '4': '18', '5': '19', 'rosterIndexWeek1': 'rosterIndexWeek3'}
roster_solution2_renamed_df = roster_solution2_df.rename(columns=renaming)[list(renaming.values())]
df = pd.concat([roster_solution1_df,roster_solution2_renamed_df], axis=1)\
    [['8', '9', '10', '11', '12','13', '14', '15', '16', '17', '18', '19',
      'rosterIndexWeek2', 'rosterIndexWeek3', 'rostersAllowedAfter', 'nurseLevel', 'nurseHours']]
df = df.assign(isCorrect=lambda x: [rosterIndex in rostersAfter for rosterIndex, rostersAfter in
                               zip(x.rosterIndexWeek3, x.rostersAllowedAfter)])


# create mip model
from ortools.sat.python import cp_model

model = cp_model.CpModel()





from contexttimer import Timer
with Timer() as t:
    df.assign(isCorrect=lambda x: [rosterIndex in rostersAfter for rosterIndex, rostersAfter in zip(x.rosterIndexWeek3, x.rostersAllowedAfter)])
print(t.elapsed)

# numbers of cores available for pandarallel
import psutil
psutil.cpu_count(logical=False)

import os
from pandarallel import pandarallel
pandarallel.initialize(nb_workers=min(os.cpu_count(), 4))

#with Timer() as t:
df['isCorrect'] = df.parallel_apply(lambda x: [row.rosterIndex in row.rostersAfter for row in x.itertuples(index=False, name=None)], axis=1)
print('bla')

df.dtypes
type(df.rostersAllowedAfter.iloc[0][0])


# testing pandarallel
import re
remove_col = "column_2"
words_to_remove_col = "column_4"

def remove_words(
    remove_from: str, words_to_remove: str, min_include_word_length: int = 4
) -> str:
    words_to_exclude = set(words_to_remove.split(" "))
    no_html = re.sub("<.*?>", " ", remove_from)
    include_words = [
        x
        for x in re.findall(r"\w+", no_html)
        if (len(x) >= min_include_word_length) and (x not in words_to_exclude)
    ]
    return " ".join(include_words)

def parapply_only_used_cols(df: pd.DataFrame, remove_col: str, words_to_remove_col: str) -> list[str]:
    return df[[remove_col, words_to_remove_col]].parallel_apply(
        lambda x: remove_words(x[remove_col], x[words_to_remove_col]), axis=1)

df1 = pd.DataFrame(
    {
        "column_1": [31, 41],
        "column_2": [
            "<p>The Apple iPhone 14, launched in 2022, comes in black, has metallic bezels and 2 or 3 cameras on the back.</p>",
            "<p>The Samsung Galaxy S22 Ultra, launched in 2022, is slim, comes in purple, has metallic bezels and multiple cameras on the back.</p>",
        ],
        "column_3": [59, 26],
        "column_4": ["Apple iPhone", "Samsung Galaxy"],
    }
)
parapply_only_used_cols(df1, remove_col, words_to_remove_col)