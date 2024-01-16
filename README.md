# pyNspPattern

Solves the NSP with week patterns

Todo
* reverse unique rosters to nurseHours-rosters, since cost will vary
Lav en “ignore shifts in period constraint”, hvilket svarer til en roster_matching med (x, -1) x er lastRosterIndex og -1 er nurseHours

Steps:
* roster_df contains unique rosters
* roster_matching is made with nurseHours
* Ignore shifts in period constraint can now be made with nurseHours=-1

Lav en must have column with list [(3, 3), (13, 0)] means that on day 3 the nurse needs to be off and on day 13 the nurse needs to have day duty.

Also have similar column with list where nurses will not have that shift. It could be [(2, 2), (4, 1)] which means on day 2 the nurse cannot have a night shift and on day 4 the nurse cannot have an evening shift.

Steps:
* Add the two columns to nurse_df with some examples like the given.
* Add nurseIndex in nurse_df and overall as an index to z in masterProblem. To be able to have unique nurses 
* Add filters to roster_df with the two column lists in the loop of the nurse_df when appending rosters to roster_indices. (Both for CG and full solution)