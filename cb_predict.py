import pandas as pd
from catboost import CatBoostClassifier

# Load data into dataframe
df = pd.read_csv('final_stats.csv')

from_file = CatBoostClassifier()
from_file.load_model("cbb_catboostmodel_02072023", format='cbm')
X = df[['Home_AdjOE', 'Home_Off_TO', 'Home_Off_OR', 'Home_Off_FTRate', 'Home_AdjDE', 'Home_Def_TO', 'Home_Def_OR', 'Home_Def_FTRate', 'Home_3P', 'Home_2P', 'Home_FT', 'Home_Tempo_Adj', 'Home_Avg_Poss_Length_Offense', 'Home_Avg_Poss_Length_Defense', 'Home_EffHgt', 'Home_Experience', 'Home_Continuity', 'Home_Best_Rank', 'Home_Top_50_Finishes', 'Home_opp_shooting_3P', 'Home_opp_shooting_2P', 'Home_opp_shooting_FT', 'Away_AdjOE', 'Away_Off_TO', 'Away_Off_OR', 'Away_Off_FTRate',
        'Away_AdjDE', 'Away_Def_TO', 'Away_Def_OR', 'Away_Def_FTRate', 'Away_3P', 'Away_2P', 'Away_FT', 'Away_Tempo_Adj', 'Away_Avg_Poss_Length_Offense', 'Away_Avg_Poss_Length_Defense', 'Away_EffHgt', 'Away_Experience', 'Away_Continuity', 'Away_Best_Rank', 'Away_Top_50_Finishes', 'Away_opp_shooting_3P', 'Away_opp_shooting_2P', 'Away_opp_shooting_FT', 'Predicted_score_Home', 'Predicted_score_away', 'Predicted_tempo', 'Predicted_spread', 'Predicted_total', 'Predicted_Result']]
df['game_prediction'] = pd.DataFrame(from_file.predict(X))
df = df[['Home/Neutral', 'Visitor/Neutral', 'Predicted_Result', 'game_prediction', 'Predicted_score_Home',
         'Predicted_score_away', 'Predicted_tempo', 'Predicted_spread', 'Predicted_total']]
