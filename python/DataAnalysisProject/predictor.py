import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

class Predictor:
    def extract_months_years(self, filename : str):
        reader = pd.read_csv(filename)
        reader['Date'] = pd.to_datetime(reader['Date'], errors = 'coerce') #convert the Date column into date time format

        reader['Year'] = reader['Date'].dt.year #create a year column
        reader['Month'] = reader['Date'].dt.month #use the reader to create a month column

        return reader #return the reader object for further functions

    def get_monthly_totals(self, reader):
        monthly_totals = reader.groupby(['Year', 'Month']).size().reset_index(name = 'Violations') #create a new pandas data frame with a new index to hold the number of violations in each year in each month
        #with pd.option_context('display.max_rows', None): #display all the rows instead of a summary
            #print(monthly_totals)
        return monthly_totals


    def make_predictions(self, totals: pd.DataFrame, month: int, year:int):
        X = totals[['Month', 'Year']] #independent variable for training the model
        y = totals['Violations'] #dependent variable for the model, number of violations in a month for a given year

        #train test split is used for evaluating the model and the accuracy of predictions. test size is the proportion of data set aside for testing versus training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state= 16) #google search: what test size for sklearn

        #https://www.youtube.com/watch?v=t3ecaDij_pU&t=418s number of estimators should be between 64-128, random_state is seed for reproducibility
        model = RandomForestRegressor(n_estimators= 110, random_state= 29 ) #https://stats.stackexchange.com/questions/36165/does-the-optimal-number-of-trees-in-a-random-forest-depend-on-the-number-of-pred
        model.fit(X_train, y_train)#train the model using the data

        model_score = model.score(X_test, y_test) #capture the model score

        input_data = pd.DataFrame({'Month': [month], 'Year': [year]}) #create a pandas data frame to pass into the model
        prediction = model.predict(input_data) #use the model to predict the number of violations, based on year and month

        #print(f"Projected number of violations for {month}/{year}: {round(prediction[0], 2)}") #print the prediction for the given month and year - debugging
        #print(f"Test performance: {round(model_score, 2)}") debugging

        return prediction[0], model_score

    def graph_predictions(self, totals: pd.DataFrame, month: int, year: int):
        X = totals[['Month', 'Year']]  # independent variable for training the model
        y = totals['Violations']  # dependent variable for the model, number of violations in a month for a given year

        # train test split is used for evaluating the model and the accuracy of predictions. test size is the proportion of data set aside for testing versus training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=16)  # google search: what test size for sklearn

        model = RandomForestRegressor(n_estimators=110, random_state=29)
        model.fit(X_train, y_train)
        violation_prediction = model.predict(X_test)

        plt.figure()
        plt.scatter(y_test, violation_prediction, color='turquoise')
        plt.xlabel("Actual Number of Violations")
        plt.ylabel("Predicted Number of Violations")
        plt.title("Actual vs Predicted Number of Violations")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


file_name = 'violationTimeOfYear.csv'
predictor = Predictor()
reader = predictor.extract_months_years(file_name)
totals = predictor.get_monthly_totals(reader)
predictor.make_predictions(totals, 12, 2040)
predictor.graph_predictions(totals, 12, 2040)

predictor = Predictor()
reader = predictor.extract_months_years(file_name)
totals = predictor.get_monthly_totals(reader)

user_month = int(input("Enter a month for violations prediction(1-12): "))
user_year = int(input("Enter a year for violations prediction: "))
result, accuracy = predictor.make_predictions(totals, user_month, user_year)
print(f"Projected number of Violations for {user_month}/{user_year}: {round(result,0)}")
print(f"Accuracy score: {round(accuracy,2)}")
