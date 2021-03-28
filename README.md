# Data Visualisation Project

## Dataset

UFC Fights between 2010 and 2021

## Files

The project is both in a Notebook and a Python file

## Column Descriptions

- R_fighter, B_fighter: Fighter names
- R_odds, B_odds: The American odds that the fighter will win. Usually scraped from bestfightodds.com
- R_ev, B_ev: The profit on a 100 credit winning bet
- date: The date of the fight
- location: The location of the fight
- country: The country the fight occurs in
- Winner: The winner of the fight [Red, Blue, or Draw]
- title_bout: Was this a title bout?
- weight_class: The weight class of the bout
- gender: Gender of the combatants
- no_of_rounds: The number of rounds in the fight
- B_current_lose_streak, R_current_lose_streak: Current losing streak
- B_current_win_streak, R_current_win_streak: Current winning streak
- B_draw, R_draw: Number of draws
- B_avg_SIG_STR_landed, R_avg_SIG_STR_landed : Significant Strikes Landed per minute
- B_avg_SIG_STR_pct, R_avg_SIG_STR_pct: Significant Striking Accuracy
- B_avg_SUB_ATT, R_avg_SUB_ATT: Average Submissions Attempted per 15 Minutes
- B_avg_TD_landed, R_avg_TD_landed: Average takedowns landed per 15 minutes
- B_avg_TD_pct, R_avg_TD_pct: Takedown accuracy
- B_longest_win_streak, R_longest_win_streak: Longest winning streak
- B_losses, R_losses: Total number of losses
- B_total_rounds_fought, R_total_rounds_fought: Total rounds fought
- B_total_title_bouts, R_total_title_bouts: Total number of title bouts
- B_win_by_Decision_Majority, R_win_by_Decision_Majority: Wins by Majority Decision
- B_win_by_Decision_Split, R_win_by_Decision_Split: Wins by Split Decision
- B_win_by_Decision_Unanimous, R_win_by_Decision_Unanimous: Wins by Unanimous Decision
- B_win_by_KO/TKO, R_win_by_KO/TKO: Wins by KO/TKO
- B_win_by_Submission, R_win_by_Submission: Wins by Submission
- B_win_by_TKO_Doctor_Stoppage, R_win_by_TKO_Doctor_Stoppage: Wins by Doctor Stoppage
- B_wins, R_wins: Total career wins
- B_Stance, R_stance: Fighter stance
- B_Height_cms, R_Height_cms: Fighter height in cms
- B_Reach_cms, R_Reach_cms: Fighter reach in cms
- B_Weight_lbs, R_Weight_lbs: Fighter weight in pounds
- B_age, R_age: Fighter age
- lose_streak_dif: (Blue lose streak) - (Red lose streak) winstreakdif: (Blue win streak) - (Red win streak)
- longest_win_streak_dif: (Blue longest win streak) - (Red longest win streak)
- win_dif: (Blue wins) - (Red wins)
- loss_dif: (Blue losses) - (Red losses)
- total_round_dif: (Blue total rounds fought) - (Red total rounds fought)
- total_title_bout_dif: (Blue number of title fights) - (Red number of title fights)
- ko_dif: (Blue wins by KO/TKO) - (Red wins by KO/TKO)
- sub_dif: (Blue wins by submission) - (Red wins by submission)
- height_dif: (Blue height) - (Red height) in cms
- reach_dif: (Blue reach) - (Red reach) in cms
- age_dif: (Blue age) - (Red age)
- sig_str_dif: (Blue sig strikes per minute) - (Red sig strikes per minute)
- avg_sub_att_dif: (Blue submission attempts) - (Red submission attempts)
- avg_td_dif: (Blue TD attempts) - (Red TD attempts)
- empty_arena: Did this fight occur in an empty arena? (1,0)
- constant_1: The number 1
- B_match_weightclass_rank, R_match_weightclass_rank: Rank in the weightclass this bout takes place in
- R_Women's Flyweight_rank, B_Women's Flyweight_rank: Rank in the Women's Flyweight Division
- B_Women's Featherweight_rank, 'RWomen's Featherweightrank: Rank in the Women's Featherweight Division BWomen's Strawweightrank, 'R_Women's Strawweight_rank: Rank in the Women's Strawweight Division
- B_Women's Bantamweight_rank, R_Women's Bantamweight_rank: Rank in the Women's Bantamweight Division
- B_Heavyweight_rank, R_Heavyweight_rank: Heavyweight rank
- B_Light Heavyweight_rank, R_Light Heavyweight rank: Light Heavyweight rank
- B_Middleweight_rank, R_Middleweight_rank: Middleweight rank
- B_Welterweight_rank, R_Welterweight_rank: Welterweight rank
- B_Lightweight_rank, R_Lightweight_rank: Lightweight rank
- B_Featherweight_rank, R_Featherweight_rank: Featherweight rank
- B_Bantamweight_rank, R_Bantamweight_rank: Bantamweight rank
- B_Flyweight_rank, R_Flyweight_rank: Flyweight rank
- B_Pound-for-Pound_rank, R_Pound-for-Pound_rank: Pound-for-Pound rank
- better_rank: Who has the better rank (Red, Blue, neither)
- finish: How the fight finished
- finish_details: More details about the finish if available.
- finish_round: The round the fight ended
- finish_round_time: Time in the round of the finish
- total_fight_time_secs: Total time of the fight in seconds

