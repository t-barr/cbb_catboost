import pandas as pd
from functools import reduce
from kenpompy.utils import login

import kenpompy.misc as kpmisc
import kenpompy.summary as kpsummary


browser = login("*****", "*****")  # browser = login("email", "password")
season = None

# Pulling KenPom stats into data frames

program_ratings = kpmisc.get_program_ratings(browser)
sch_strength = kpmisc.get_pomeroy_ratings(browser, season)
tempo = kpsummary.get_efficiency(browser, season)
four_factors = kpsummary.get_fourfactors(browser, season)
shooting = kpsummary.get_teamstats(browser, defense=False, season=None)
opp_shooting = kpsummary.get_teamstats(browser, defense=True, season=None)
team_characteristics = kpsummary.get_height(browser, season)

# Dropping unwanted columns

team_characteristics.drop(team_characteristics.columns[[
                          1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 21]], axis=1, inplace=True)
opp_shooting.drop(opp_shooting.columns[[
                  1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]], axis=1, inplace=True)
shooting.drop(shooting.columns[[1, 3, 5, 7, 8, 9, 10,
              11, 12, 13, 14, 15, 16, 17]], axis=1, inplace=True)
four_factors.drop(four_factors.columns[[
                  2, 3, 5, 6, 7, 9, 11, 13, 15, 16, 17, 19, 21, 23]], axis=1, inplace=True)
tempo.drop(tempo.columns[[1, 3, 4, 5, 7, 9, 10, 11, 12,
           13, 14, 15, 16, 17]], axis=1, inplace=True)
sch_strength.drop(sch_strength.columns[[
                  0, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 15, 16, 17, 18, 21]], axis=1, inplace=True)
program_ratings.drop(program_ratings.columns[[
                     0, 2, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16]], axis=1, inplace=True)

# Creating CSV file of each data frame

program_ratings.to_csv('program_ratings.csv')
sch_strength.to_csv('sch_strength.csv')
tempo.to_csv('tempo.csv')
four_factors.to_csv('four_factors.csv')
shooting.to_csv('shooting.csv')
opp_shooting.to_csv('opp_shooting.csv')
team_characteristics.to_csv('team_characteristics.csv')
