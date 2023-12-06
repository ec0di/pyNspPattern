from ortools.linear_solver import pywraplp
import time
from contexttimer import Timer

def model_master(n_days, n_work_shifts, nurse_df, roster_df, binary_plans, demand, t_max_sec, solver_id='GLOP'):  # solver_id='CBC' for MIP or GLOP for LP
    solver = pywraplp.Solver.CreateSolver(solver_id=solver_id)

    start_time = time.time()
    #roster_indices = [np.arange(len(rosters[nurseType])) for nurseType in nurseTypes]

    # Decision Variables
    with Timer() as t:
        z = {}
        for nurse_type in nurse_df.nurseType.unique():
            for roster_idx in roster_df[roster_df.nurseType == nurse_type].rosterIndex.values:
                if solver_id == 'GLOP':
                    z[nurse_type, roster_idx] = solver.NumVar(name=f'z_{nurse_type},{roster_idx}', lb=0, ub=float(nurse_df['nurseCount'].sum()))
                else: # solver_id == 'CBC'
                    z[nurse_type, roster_idx] = solver.IntVar(name=f'z_{nurse_type},{roster_idx}', lb=0, ub=float(nurse_df['nurseCount'].sum()))
        print(f'Decision variables created and took: {t.elapsed}')
    #####################
    # Constraints
    nurse_c = dict()  # originally nurse_counts
    for nurse_type, nurse_count in nurse_df[['nurseType', 'nurseCount']].itertuples(index=False):
        nurse_c[nurse_type] = solver.Add(solver.Sum([z[nurse_type, roster_idx]
                                                     for roster_idx in roster_df[roster_df.nurseType == nurse_type].rosterIndex.values])
                                         == nurse_count, name=f"nurse_{nurse_type}")

    # demand_c = dict()
    # for j in range(n_days):
    #     for k in range(n_work_shifts):
    #         demand_c[(j, k)] = solver.Add(solver.Sum([z[nurseType, roster_idx] * binary_plans[roster_idx][j, k]
    #                                                   for nurseType in nurse_df.nurseType.values
    #                                                   for roster_idx in roster_df[roster_df.nurseType == nurseType].rosterIndex.values])
    #                                       >= demand[k, j], name=f"demand_{j}_{k}")

    demand_c = {(j, k): solver.Add(solver.Sum([z[nurse_type, roster_idx] * binary_plans[roster_idx][j, k]
                                                      for nurse_type in nurse_df.nurseType.values
                                                      for roster_idx in roster_df[roster_df.nurseType == nurse_type].rosterIndex.values])
                                          >= demand[k, j], name=f"demand_{j}_{k}")
                for j in range(n_days) for k in range(n_work_shifts)}

    # comp level demand constraint
    # demand_comp_level_c = dict()
    # demand_nurse_level = 1
    # for j in range(n_days):
    #     for k in range(n_work_shifts):
    #         demand_comp_level_c[(j, k)] = solver.Add(solver.Sum([z[nurseType, roster_idx] * binary_plans[roster_idx][j, k]
    #                                                   for nurseType in nurse_df[nurse_df['nurseLevel']==3].rosterIndex.values
    #                                                   for roster_idx in
    #                                                   roster_df[roster_df.nurseType == nurseType].rosterIndex.values])
    #                                       >= demand_nurse_level, name=f"demand_{j}_{k}")
    demand_advanced_nurse_level = 1
    demand_advanced_nurse_level_c = {(j, k): solver.Add(solver.Sum([z[nurse_type, roster_idx] * binary_plans[roster_idx][j, k]
                                                         for nurse_type in nurse_df[nurse_df['nurseLevel'] == 3].nurseType.values
                                                         for roster_idx in roster_df[roster_df.nurseType == nurse_type].rosterIndex.values])
                                             >= demand_advanced_nurse_level, name=f"demand_advanced_nurse_level_{j}_{k}") for j in range(n_days) for k in range(n_work_shifts)}

    # todo, investigate rosters
    #extra = 0
    #for j in range(n_days):
    #    for k in range(n_work_shifts):
    #        if k in [1, 2, 4, 5] or np.mod(j, 7) > 4:
    #            extra += solver.Sum([z[nurseType, roster_idx] * rosters[nurseType][roster_idx][0][j, k]
    #                                 for nurseType in nurseTypes
    #                                 for roster_idx in roster_indices[nurseType]]) - demand[k, j]
    obj = solver.Sum([z[nurse_type, roster_idx] * total_cost for nurse_type in nurse_df.nurseType.values
                      for roster_idx, total_cost in roster_df[roster_df.nurseType == nurse_type][['rosterIndex', 'totalCost']].itertuples(index=False)])# + extra * 1

    setup_time = time.time()

    print(f'Time to setup problem: {round(setup_time - start_time, 2)} s')

    if solver_id in ['CBC', 'SCIP']:
        solver.SetTimeLimit(int(1000 * t_max_sec))
    solver.Minimize(obj)
    status = solver.Solve()

    print(f'Time to solve: {round(time.time() - setup_time, 2)} s')

    return solver, nurse_c, demand_c, demand_advanced_nurse_level_c, z, status
