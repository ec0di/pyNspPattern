from ortools.linear_solver import pywraplp
import time
from contexttimer import timer


def master_problem_instance(n_days, n_work_shifts, nurse_df, roster_indices, roster_costs, binary_plans, demand,
                            max_time_solver_sec, solver_id='GLOP', verbose=True):  # GLOP is LP, CBC is MIP
    solver = pywraplp.Solver.CreateSolver(solver_id=solver_id)

    start_time = time.time()

    z = create_decision_variables(nurse_df, roster_indices, solver, solver_id)

    # Constraints
    nurse_c = n_rosters_must_match_nurse_count_constraint(nurse_df, roster_indices, solver, z)

    demand_c = all_nurses_demand_constraint(binary_plans, demand, n_days, n_work_shifts, nurse_df, roster_indices, solver, z)

    demand_advanced_nurse_level = 1
    demand_advanced_nurse_level_c = advanced_nurses_demand_constraint(binary_plans, demand_advanced_nurse_level, n_days,
                                                                      n_work_shifts, nurse_df, roster_indices, solver, z)

    obj = create_objective(nurse_df, roster_indices, roster_costs, solver, z)

    setup_time = time.time()

    if verbose:
        print(f'Time to setup problem: {round(setup_time - start_time, 2)} s')

    if solver_id in ['CBC', 'SCIP']:
        solver.SetTimeLimit(int(1000 * max_time_solver_sec))
    solver.Minimize(obj)
    status = solver.Solve()

    if verbose:
        print(f'Time to solve: {round(time.time() - setup_time, 2)} s')
        if solver_id == 'CBC':
            print(f"Integer Solution with Object {round(solver.Objective().Value(), 1)}")
        elif solver_id == 'GLOP':
            print(f"LP Solution with Object {round(solver.Objective().Value(), 1)}")

    return solver, status, demand_c, z


#@timer()
def create_objective(nurse_df, roster_indices, roster_costs, solver, z):
    obj = solver.Sum([z[nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index, roster_idx] * roster_costs[roster_idx]
                      for nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index in
                        nurse_df[['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']].itertuples(index=False)
                      for roster_idx in roster_indices[nurse_hours, last_one_week_roster_index]])
    return obj


#@timer()
def advanced_nurses_demand_constraint(binary_plans, demand_advanced_nurse_level, n_days, n_work_shifts, nurse_df,
                                      roster_indices, solver, z):
    demand_advanced_nurse_level_c = {
        (j, k): solver.Add(solver.Sum([z[nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index, roster_idx] * binary_plans[roster_idx][j, k]
                                       for nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index in
                                        nurse_df[nurse_df['nurseLevel'] == 3][['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']].itertuples(index=False)
                                       for roster_idx in roster_indices[nurse_hours, last_one_week_roster_index]])
                           >= demand_advanced_nurse_level, name=f"demand_advanced_nurse_level_{j}_{k}") for j in
        range(n_days) for k in range(n_work_shifts)}
    return demand_advanced_nurse_level_c


#@timer()
def all_nurses_demand_constraint(binary_plans, demand, n_days, n_work_shifts, nurse_df, roster_indices, solver, z):
    demand_c = {(j, k): solver.Add(solver.Sum([z[nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index, roster_idx] * binary_plans[roster_idx][j, k]
                                               for nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index in
                                               nurse_df[['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']].itertuples(index=False)
                                               for roster_idx in roster_indices[nurse_hours, last_one_week_roster_index]])
                                   >= demand[k, j], name=f"demand_{j}_{k}")
                for j in range(n_days) for k in range(n_work_shifts)}
    return demand_c


#@timer()
def n_rosters_must_match_nurse_count_constraint(nurse_df, roster_indices, solver, z):
    nurse_c = dict()
    for nurse_hours, nurse_level, nurse_count, last_one_week_roster_index, last_two_week_roster_index in \
        nurse_df[['nurseHours', 'nurseLevel', 'nurseCount', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']].itertuples(index=False):
        nurse_c[nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index] = \
            solver.Add(solver.Sum([z[nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index, roster_idx]
                for roster_idx in roster_indices[nurse_hours, last_one_week_roster_index]])
                    == nurse_count, name=f"nurse_{nurse_hours}_{nurse_level}_{last_one_week_roster_index}_{last_two_week_roster_index}")
    return nurse_c


#@timer()
def create_decision_variables(nurse_df, roster_indices, solver, solver_id):
    z = {}
    for nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index in \
        nurse_df[['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'lastTwoWeekRosterIndex']].itertuples(index=False):
        for roster_idx in roster_indices[nurse_hours, last_one_week_roster_index]:
            if solver_id == 'GLOP':
                z[nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index, roster_idx] = \
                    solver.NumVar(name=f'z_{nurse_hours}_{nurse_level}_{last_one_week_roster_index}_{last_two_week_roster_index}_{roster_idx}',
                                  lb=0, ub=float(nurse_df['nurseCount'].sum()))
            else:  # solver_id == 'CBC'
                z[nurse_hours, nurse_level, last_one_week_roster_index, last_two_week_roster_index, roster_idx] = \
                    solver.IntVar(name=f'z_{nurse_hours}_{nurse_level}_{last_one_week_roster_index}_{last_two_week_roster_index}_{roster_idx}',
                                  lb=0, ub=float(nurse_df['nurseCount'].sum()))
    return z
