# pyNspPattern

Solves the NSP with week patterns

Todo
* reverse unique rosters to nurseHours-rosters, since cost will vary

* Lav en must have column with list [(3, 3), (13, 0)] means that on day 3 the nurse needs to be off and on day 13 the nurse needs to have day duty.

* Also have similar column with list where nurses will not have that shift. It could be [(2, 2), (4, 1)] which means on day 2 the nurse cannot have a night shift and on day 4 the nurse cannot have an evening shift.

Steps:
* Add the two columns to nurse_df with some examples like the given.
* Add nurseIndex in nurse_df and overall as an index to z in masterProblem. To be able to have unique nurses 
* Add filters to roster_df with the two column lists in the loop of the nurse_df when appending rosters to roster_indices. (Both for CG and full solution)

Another thing:
* Make nurse=-1 in roster_df where all plans are present from workShifts=0..10, make roster_matching 
on this group and make this group the fallback group if a nurse_group does not have any plans left in roster_df.
Wait with this one.