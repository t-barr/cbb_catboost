import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data into dataframe
df = pd.read_csv('final_stats.csv')

# Split into data (X) and training sets (y)
X = df[['Home_AdjOE', 'Home_Off_TO', 'Home_Off_OR', 'Home_Off_FTRate', 'Home_AdjDE', 'Home_Def_TO', 'Home_Def_OR', 'Home_Def_FTRate', 'Home_3P', 'Home_2P', 'Home_FT', 'Home_Tempo_Adj', 'Home_Avg_Poss_Length_Offense', 'Home_Avg_Poss_Length_Defense', 'Home_EffHgt', 'Home_Experience', 'Home_Continuity', 'Home_Best_Rank', 'Home_Top_50_Finishes', 'Home_opp_shooting_3P', 'Home_opp_shooting_2P', 'Home_opp_shooting_FT', 'Away_AdjOE', 'Away_Off_TO', 'Away_Off_OR', 'Away_Off_FTRate',
        'Away_AdjDE', 'Away_Def_TO', 'Away_Def_OR', 'Away_Def_FTRate', 'Away_3P', 'Away_2P', 'Away_FT', 'Away_Tempo_Adj', 'Away_Avg_Poss_Length_Offense', 'Away_Avg_Poss_Length_Defense', 'Away_EffHgt', 'Away_Experience', 'Away_Continuity', 'Away_Best_Rank', 'Away_Top_50_Finishes', 'Away_opp_shooting_3P', 'Away_opp_shooting_2P', 'Away_opp_shooting_FT', 'Predicted_score_Home', 'Predicted_score_away', 'Predicted_tempo', 'Predicted_spread', 'Predicted_total', 'Predicted_Result']]
y = df['Result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.18)

# Train catboost model
model = CatBoostClassifier(
    iterations=500, max_depth=10, learning_rate=.0018, random_seed=42, )
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Save model
model.save_model("cbb_catboostmodel_02072023")
