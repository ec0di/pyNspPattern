from master_problem import master_problem_instance
from helpers import calculate_parameters, set_dataframe_print_width, \
    write_to_parquet, hotfix_for_pandas_merge, ColumnGenerationParameters
from input_parameters import SHIFT_LENGTH_IN_HOURS, N_WORK_SHIFTS, BASE_DEMAND, nurse_df, HARD_SHIFTS_FAIR_PLANS_FACTOR, \
    WEEKEND_SHIFTS_FAIR_PLAN_FACTOR
from roster_factory import RosterFactory
from plots import visualize_optimized_nurse_schedule, visualize_optimized_nurse_demand_surplus

import numpy as np
import pandas as pd
import os
pd.options.plotting.backend = "plotly"


class NurseScheduler:

    def __init__(self,
                 n_weeks: int,
                 use_column_generation: bool = False,
                 nurse_df: pd.DataFrame = None,
                 pct_of_best_rosters_to_keep: float = 1.0,
                 cg_parameters: ColumnGenerationParameters = None,
                 solution_base_path: str = None,
                 verbose: bool = True):
        self.n_weeks = n_weeks
        self.use_column_generation = use_column_generation
        self.nurse_df = nurse_df
        self.pct_of_best_rosters_to_keep = pct_of_best_rosters_to_keep
        self.cg_parameters = cg_parameters
        self.solution_base_path = solution_base_path
        self.verbose = verbose

        self.n_days = n_weeks * 7

        self.one_week_roster_file = 'data/1WeekRosters.parquet'
        self.two_week_roster_file = 'data/2WeekRosters.parquet'
        self.roster_matching_file = 'data/1WeekRosterMatching.json'

    def ensure_setup_files_are_available(self, overwrite_setup_files=False):
        print('Running Setup if necessary')
        if not all([os.path.exists(self.one_week_roster_file), os.path.exists(self.roster_matching_file)]) or overwrite_setup_files:
            print(f'Creating file {self.one_week_roster_file}')
            self.write_setup_files(1)

        if self.n_weeks >= 2 and (not os.path.exists(self.two_week_roster_file) or overwrite_setup_files):
            print(f'Creating file {self.two_week_roster_file}')
            self.write_setup_files(2)
        print('Setup is ready')
        print('-----------------------------------')

    def write_setup_files(self, n_weeks):
        cost_parameters, feasibility_parameters = calculate_parameters(n_weeks, N_WORK_SHIFTS, self.nurse_df, BASE_DEMAND,
                                                                       HARD_SHIFTS_FAIR_PLANS_FACTOR,
                                                                       WEEKEND_SHIFTS_FAIR_PLAN_FACTOR,
                                                                       SHIFT_LENGTH_IN_HOURS)

        roster_factory = RosterFactory(n_weeks, N_WORK_SHIFTS, self.nurse_df, cost_parameters, feasibility_parameters)

        roster_df = roster_factory.calculate_roster_df()
        roster_df.columns = [str(colname) for colname in roster_df.columns]

        if n_weeks == 1:
            roster_factory.calculate_roster_matching()
            roster_factory.write_roster_matching(self.roster_matching_file)

        # create 2 week roster df with 1week rosters matching
        if n_weeks >= 2:
            roster_df = roster_factory.append_one_week_roster_index_to_two_week_roster_df(self.one_week_roster_file)

        print(f'Exporting file data/{n_weeks}WeekRosters.parquet')
        roster_df.to_parquet(f'data/{n_weeks}WeekRosters.parquet', index=False)

    def optimized_nurse_schedule_iteration(self, n_weeks_in_iteration, demand, roster_factory: RosterFactory,
                                           solution_path: str = None):
        if self.use_column_generation:
            n_largest_for_each_nurse = 3  # necessary with 3 to get full 0s, 1s, and 2s plans
            n_smallest_for_each_nurse = 5 ** n_weeks_in_iteration
            roster_factory.run_initial_solution_for_cg(n_largest_for_each_nurse, n_smallest_for_each_nurse)

            roster_factory.run_column_generation(verbose=self.verbose,
                                                 demand=demand,
                                                 max_time_sec=self.cg_parameters.max_time_sec,
                                                 max_iter=self.cg_parameters.max_iter,
                                                 min_object_value=15,
                                                 n_rosters_per_iteration=self.cg_parameters.n_rosters_per_iteration,
                                                 solver_id='GLOP',
                                                 max_time_per_iteration_sec=self.cg_parameters.max_time_per_iteration_sec, )

        else:  # full set of rosters solution
            roster_factory.run_full_solution_for_mip()

        # run final master problem with MIP
        solver, status, demand_c, z = master_problem_instance(n_weeks=n_weeks_in_iteration, n_work_shifts=N_WORK_SHIFTS,
                                                              nurse_df=roster_factory.nurse_df,
                                                              roster_indices=roster_factory.roster_indices,
                                                              roster_costs=roster_factory.roster_costs,
                                                              binary_plans=roster_factory.binary_plans,
                                                              demand=demand,
                                                              max_time_solver_sec=1000, solver_id='CBC')
        self.write_out_solution(n_weeks_in_iteration, roster_factory, z, status, solution_path)

    def calculate_optimized_nurse_schedule(self, overwrite_setup_files=False):

        self.ensure_setup_files_are_available(overwrite_setup_files=overwrite_setup_files)

        # demand and supply of nurses
        print('Nurse demand vs supply per week: ', sum(sum(BASE_DEMAND)), 'vs', sum(self.nurse_df.nurseHours / SHIFT_LENGTH_IN_HOURS * self.nurse_df.nurseCount))
        print('Preparing Data')
        n_weeks_in_iteration = 2 if self.n_weeks >= 2 else 1
        demand = np.tile(BASE_DEMAND, (1, n_weeks_in_iteration))  # D, 98 in total per week

        # quick check of demand vs supply
        cost_parameters, feasibility_parameters = calculate_parameters(n_weeks_in_iteration, N_WORK_SHIFTS, self.nurse_df, BASE_DEMAND,
                                                                       HARD_SHIFTS_FAIR_PLANS_FACTOR,
                                                                       WEEKEND_SHIFTS_FAIR_PLAN_FACTOR,
                                                                       SHIFT_LENGTH_IN_HOURS)

        roster_factory = RosterFactory(n_weeks_in_iteration, N_WORK_SHIFTS, self.nurse_df, cost_parameters, feasibility_parameters)
        roster_factory.read_roster_df_from_parquet(f'data/{n_weeks_in_iteration}WeekRosters.parquet')
        if n_weeks >= 2:
            roster_factory.filter_roster_df(self.pct_of_best_rosters_to_keep)
            roster_factory.load_roster_matching(self.roster_matching_file)

        # create map of day, work_shifts to rosters
        roster_factory.append_day_work_shift_flags()
        print('-----------------------------------')

        if self.n_weeks >= 2:
            for n_weeks_iteration in range(0, self.n_weeks//2):
                i = n_weeks_iteration * 2
                print(f'Creating plan for week {i} to {i + 1}')
                solution_path = f'{self.solution_base_path}Week{i}to{i + 1}.parquet'
                if n_weeks_iteration >= 1:
                    # create nurse_df from solution file
                    roster_solution_df = pd.read_parquet(f'{self.solution_base_path}Week{i-2}to{i-1}.parquet')
                    nurse_df = roster_solution_df.assign(lastOneWeekRosterIndex=lambda x: x.rosterIndexWeek2,
                                                         twoWeekRosterIndexHistory=lambda x: x.twoWeekRosterIndexHistory + ', ' + x.rosterIndex.astype(str)).\
                        groupby(['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'twoWeekRosterIndexHistory']).\
                        agg(nurseCount=('nRostersInSolution', 'sum')).reset_index().astype({'nurseCount': 'int32'})
                    self.nurse_df = nurse_df
                    print(nurse_df.nurseCount.sum())
                nurse_df = self.nurse_df.assign(nurseShifts=lambda x: x.nurseHours // SHIFT_LENGTH_IN_HOURS * n_weeks_in_iteration)

                roster_factory.nurse_df = nurse_df
                self.optimized_nurse_schedule_iteration(n_weeks_in_iteration, demand, roster_factory, solution_path)
                print('-----------------------------------')

    def write_out_solution(self, n_weeks, roster_factory, z, status, solution_path):
        if status == 0:  # optimal solution found
            z = {key: value.solution_value() for key, value in z.items()}
            r_indices_df = pd.DataFrame(list(keys)+[value] for keys, value in z.items() if value >= 1)
            r_indices_df.columns = ['nurseHours', 'nurseLevel', 'lastOneWeekRosterIndex', 'twoWeekRosterIndexHistory', 'rosterIndex', 'nRostersInSolution']

            r_indices_df, roster_factory.roster_df = hotfix_for_pandas_merge(r_indices_df, roster_factory.roster_df)
            roster_solution_df = roster_factory.roster_df.merge(r_indices_df, how='inner', on=['nurseHours', 'rosterIndex'])
            if self.verbose:
                print(roster_solution_df.loc[:, [str(x) for x in np.arange(n_weeks * 7)]])
            write_to_parquet(roster_solution_df, solution_path)


if __name__ == "__main__":
    n_weeks = 6
    assert n_weeks % 2 == 0, 'n_weeks must be an even number'
    use_column_generation = True
    verbose = False
    solution_base_path = f'data/{n_weeks}WeekRosterSolution'

    # parameters
    pct_of_best_rosters_to_keep = 0.25

    # column generation parameters
    max_time_sec = 5
    max_time_per_iteration_sec = 100
    max_iter = 1000
    n_rosters_per_iteration = 300
    column_generation_parameters = ColumnGenerationParameters(max_time_sec, max_time_per_iteration_sec, max_iter,
                                                              n_rosters_per_iteration)

    nurse_scheduler = NurseScheduler(n_weeks=n_weeks,
                                     use_column_generation=use_column_generation,
                                     nurse_df=nurse_df,
                                     pct_of_best_rosters_to_keep=pct_of_best_rosters_to_keep,
                                     cg_parameters=column_generation_parameters,
                                     solution_base_path=solution_base_path,
                                     verbose=verbose)
    nurse_scheduler.calculate_optimized_nurse_schedule(overwrite_setup_files=True)

    # visualize results
    fig1 = visualize_optimized_nurse_schedule(solution_base_file=solution_base_path, n_weeks=n_weeks, explode_nurse_count=False)
    fig2 = visualize_optimized_nurse_demand_surplus(solution_base_file=solution_base_path, n_weeks=n_weeks)

    # for exploring dataframes
    set_dataframe_print_width()
