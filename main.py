import numpy as np
import pandas as pd

from master_problem import master_problem
from helpers import get_demand, calculate_parameters, read_rosters_from_parquet
from roster_factory import calculate_roster_df
from partial_roster import PartialRoster


read_solution = True
n_weeks = 2  # works for 1 week with nurse_type from bla

n_work_shifts = 3
n_days = n_weeks * 7
base_path = ''
parquet_filename = f'{base_path}data/{n_weeks}WeekRosters.parquet'

# demand per week
base_demand = np.array([[3, 4, 3, 4, 3, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2]])
demand = get_demand(base_demand, n_weeks)

# read in solution
nurse_df_base = pd.read_excel(base_path + 'data/NurseData.xlsx', sheet_name="personindstillinger")
nurse_df = nurse_df_base.groupby(['nurseHours', 'nurseLevel']).agg(nurseCount=('Person', 'count')).reset_index()\
    .rename_axis('nurseType').reset_index()

# factors
hard_shifts_fair_plans_factor = 0.5
weekend_shifts_fair_plan_factor = 0.5

cost_parameters, feasibility_parameters = calculate_parameters(n_weeks, n_work_shifts, nurse_df, base_demand,
                                                               hard_shifts_fair_plans_factor,
                                                               weekend_shifts_fair_plan_factor)

if read_solution:
    roster_df, binary_plans = read_rosters_from_parquet(parquet_filename, n_days, n_work_shifts)

else:
    roster_df, binary_plans = calculate_roster_df(nurse_df, n_days, n_work_shifts, cost_parameters, feasibility_parameters)

    # write out solution
    roster_df.columns = [str(colname) for colname in roster_df.columns]  # write df to parquet
    roster_df.to_parquet(parquet_filename, index=False)

# run optimization problem
#roster_df = pd.read_csv(csv_filename)


#nurse_df = nurse_df_base.groupby(['nurseHours', 'nurseLevel']).agg(nurseCount=('Person', 'count')).reset_index()

#nurse_df = pd.DataFrame({'nurseHours': [37, 37], 'nurseLevel': [1, 3], 'nurseCount': [6, 12], 'nurseType': [4, 5]})  # works!
#nurse_df = pd.DataFrame({'nurseHours': [37], 'nurseLevel': [3], 'nurseCount': [12], 'nurseType': [4]})  # works!
#solver, nurse_c, demand_c, z, status = model_master(n_days, n_work_shifts, nurse_df, roster_df, binary_plans, demand, t_max_sec=10, solver_id='GLOP')

#top_n_pct = 50
#roster_df_ = roster_df.sort_values('totalCost').iloc[0:round(roster_df.shape[0]/100*top_n_pct)]

#n_smallest = 600
#roster_df_ = roster_df.merge(roster_df.rename_axis('rosterIndex').groupby('nurseType')['totalCost'].nsmallest(n_smallest).reset_index().drop(columns=['totalCost']),
#                             how='inner', on=['nurseType', 'rosterIndex'])

# todo, make initial solution simple

solver, nurse_c, demand_c, demand_comp_level_c, z, status = master_problem(n_days, n_work_shifts, nurse_df, roster_df,
                                                                           binary_plans, demand,
                                                                           t_max_sec=300, solver_id='GLOP')  # CBC, GLOP
# sub problem iterations where we find and add rosters

obj_int = solver.Objective().Value()
print(f"Integer Solution with Object {round(obj_int, 1)}")

if status == 0:
    z_int = {key: value.solution_value() for key, value in z.items()}

    r_indices_df = pd.DataFrame([key[0],key[1], value] for key, value in z_int.items() if value >= 1)
    r_indices_df.columns = ['nurseType', 'rosterIndex', 'nRostersInSolution']

    # nice to remember: .loc[lambda x: x.nurseType == 2]
    roster_solution_df = roster_df.merge(r_indices_df, how='inner', on=['rosterIndex', 'nurseType'])
    print(roster_solution_df.loc[:, [str(x) for x in np.arange(n_days)]])


# patch 2 rosters together
cost_parameters, feasibility_parameters = calculate_parameters(2 * n_weeks, n_work_shifts, nurse_df, base_demand,
                                                               hard_shifts_fair_plans_factor, weekend_shifts_fair_plan_factor)

base_roster = PartialRoster(n_days=2 * n_days,
                            nurse_type=5,
                            nurse_hours=37,
                            n_work_shifts=n_work_shifts,
                            cost_parameters=cost_parameters,
                            feasibility_parameters=feasibility_parameters)

if False:
    import time
    start_time = time.time()

    finished_patched_rosters = []
    for roster1 in finished_rosters:
        for roster2 in finished_rosters:
            new_roster = copy.deepcopy(base_roster)
            for shift in roster1.plan + roster2.plan:
                feasible_shifts = new_roster.feasible_shifts()
                if shift not in feasible_shifts:
                    #print(f'break at {new_roster.plan} and increment with {shift}')
                    break
                new_roster.increment(shift)
            if new_roster.is_finished():
                finished_patched_rosters.append(new_roster)

    print(f'Patching together rosters took: {round(time.time() - start_time, 2)} s')