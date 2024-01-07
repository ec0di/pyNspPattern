import numpy as np
import pandas as pd
import json
pd.set_option('display.max_columns', None)

import numpy as np

desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)

from master_problem import master_problem_instance
from helpers import get_demand, calculate_parameters
from roster_factory import RosterFactory
from partial_roster import PartialRoster

base_path = ''

roster_solution1_df = pd.read_parquet(f'{base_path}data/2RosterSolutionStatic.parquet')
nurse1_df = roster_solution1_df.rename(columns={'rosterIndexWeek2': 'lastRosterIndex'}).\
    groupby(['nurseHours', 'nurseLevel', 'lastRosterIndex']).\
    agg(nurseCount=('nRostersInSolution', 'sum')).reset_index().astype({'nurseCount': 'int32'})

roster_solution2_df = pd.read_parquet(f'{base_path}data/2RosterSolution.parquet')
nurse2_df = roster_solution2_df.rename(columns={'rosterIndexWeek2': 'lastRosterIndex'}).\
    groupby(['nurseHours', 'nurseLevel', 'lastRosterIndex']).\
    agg(nurseCount=('nRostersInSolution', 'sum')).reset_index().astype({'nurseCount': 'int32'})

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