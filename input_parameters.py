import numpy as np
import pandas as pd


# Case specific parameters
OFF_SHIFT = 3
MAX_CONSECUTIVE_WORK_SHIFTS = 5
DELTA_NURSE_SHIFT = 1
SHIFT_LENGTH_IN_HOURS = 8
N_WORK_SHIFTS = 3

BASE_DEMAND = np.array([[3, 4, 3, 4, 3, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2],
                        [2, 2, 2, 2, 2, 2, 2]])
NURSE_DF_MULTIPLIER = 4
BASE_DEMAND *= NURSE_DF_MULTIPLIER

nurse_df = pd.DataFrame({'nurseHours': [28, 28, 32, 32, 37, 37],
                         'nurseLevel': [1, 3, 1, 3, 1, 3],
                         'nurseCount': [1, 1, 1, 4, 4, 3]})
nurse_df['lastOneWeekRosterIndex'] = -1  # means all rosters are available
nurse_df['twoWeekRosterIndexHistory'] = '-1'  # means all rosters are available
nurse_df.nurseCount *= NURSE_DF_MULTIPLIER

COSTS = {
    'consecutiveShifts': -0.04,
    'missingTwoDaysOffAfterNightShifts': 0.1,
    'moreThanTwoConsecutiveNightShifts': 1,
    'singleNightShift': 1,
    'moreThanFourConsecutiveWorkShifts': 1,
    'afternoonShiftsFair': None,
    'nightShiftsFair': None,
    'nightAndAfternoonShiftsFair': None,
    'weekendShiftsFair': None}

# fair plan factors
HARD_SHIFTS_FAIR_PLANS_FACTOR = 0.5
WEEKEND_SHIFTS_FAIR_PLAN_FACTOR = 0.5
