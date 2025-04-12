import numpy as np #library designed for numerical and scientific computing.
import matplotlib.pyplot as plt # Simple and intuitive to create visuals by combining functions for plotting
import pandas as pd # Library widely used for data manipulation, analysis, and cleaning.
from sklearn.model_selection import train_test_split as tts # library for splitting data, tuning hyperparameters. 
from sklearn.linear_model import LinearRegression # Linear Regression for simple and multiple regression problems.
from sklearn.metrics import r2_score as rscr # Library for is essential for evaluating model performance. 
import pickle # Library for save and load specific objects!
import warnings # Suppress the warning 

warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

regr = LinearRegression() # Assign function to a variable.
# Load dataset
data = pd.read_csv(r'50_Startups.csv') # Load input value into memory
data = pd.get_dummies(data,drop_first=True) # Load input value into memory

# Independent and dependent variables
indep = data[['R&D Spend', 'Administration', 'Marketing Spend', 'State_Florida', 'State_New York']] # Loar independent value into variable
dep = data[['Profit']] # Loar dependent value into variable

# # Scatter plot
# plt.scatter(indep[['R&D Spend']], dep, label="R&D Spend", color="blue")
# plt.scatter(indep[['Administration']], dep, label="Administration", color="red")
# plt.scatter(indep[['Marketing Spend']], dep, label="Marketing Spend", color="green")
# plt.xlabel("Spend", fontsize=20)
# plt.ylabel("Profit", fontsize=20)
# plt.show()  # Uncommented to show the plot

# Split the data into training and testing sets 
x_train, x_test, y_train, y_test = tts(indep, dep, test_size=1/3, random_state=0)
regr.fit(x_train, y_train)

# # Get the model's weights (coefficients) and intercept
weight = regr.coef_
bais = regr.intercept_
print([[weight]])
print(f"Weight of the model = {float(weight[0][0]):,.2f}, {float(weight[0][1]):,.2f},{float(weight[0][2]):,.2f}, {float(weight[0][3]):,.2f}, {float(weight[0][4]):,.2f}")
print(f"Intercept of the model = {float(bais[0]):,.2f}")

# # Predict using the trained model
y_pred = regr.predict(x_test)

# # Calculate and print R-squared score
rscore = rscr(y_test, y_pred)
print(f"Prediction accuracy: {rscore*100:.2f} %")

# #if Good model export the file 
if (rscore > 0.85):
    fn = "multi-final.sav"
    pickle.dump(regr,open(fn,'wb'))
    #import the file for testing 
    lmlp  = pickle.load(open(fn,"rb"))
    print ("Good Model")
    print()
    print()
    print("* * * Test for the imported model * * *")
    respo = lmlp.predict([[93863.75,127320.38,249839.44,0,0]]) # calling imorted module
    print(f"California Profit = {float(respo[0][0]):,.2f}") 
    print()
    respo = lmlp.predict([[93863.75,127320.38,249839.44,1,0]]) # calling imorted module
    print(f"Florida Profit = {float(respo[0][0]):,.2f}") 
    print()
    respo = lmlp.predict([[93863.75,127320.38,249839.44,0,1]]) # calling imorted module
    print(f"New York = {float(respo[0][0]):,.2f}") 
    print()
else: 
    print ("Bad Model")


