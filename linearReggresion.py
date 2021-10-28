import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

#Pobranie danych z pliku csv
bitcoin = pd.read_csv('C:/Users/lukas/Desktop/Programowanie/Magisterka/BTC-USD-Month.csv')
bitcoin = bitcoin.dropna() #wartości null zostaną usunięte


#Wydruk wartości pliku zapisanego w zmiennej bitcoin
#print(bitcoin)

#Sprawdzenie zestawu danych, czy zawiera kilka nieznanych wartości oraz zostaną one usunięte przez "dropna()"
bitcoin.isnull().sum()
bitcoin.dropna(inplace=True)


#Wybranie danych jakie będą urzyte do uczenia, a które do testowania:
#15% - dane do testów
#85% - dane do uczenia
X = ["Open", "High", "Low", "Volume"]
Y = "Close"

x_train, x_test, y_train, y_test = train_test_split(
bitcoin[X],
bitcoin[Y],
test_size = .15,
random_state=0
)

#Model regresji liniowej
model = LinearRegression()
model.fit(x_train, y_train)
#Score, czyli sprawdzam jak model radzi sobie na danych testowych, czy dobrze sobie poradził z przewidywaniem cen
print('Score: ' ,model.score(x_test, y_test))

#Predict prices:
#Predykcja na 7 dni
future_set = bitcoin.shift(periods = 0).tail(7)
print(future_set)
prediction = model.predict(future_set[X])

#Wydruk danych porównawczych w tabeli(Realne i Przewidziane):
df = pd.DataFrame({'Real Values' :future_set['Close'], 'Predicted Values' :prediction})
print(df)

#plt.plot(bitcoin["Date"][-100:-29], bitcoin["Close"][-100:-29], color='goldenrod', lw=2)
plt.scatter(bitcoin["Date"], bitcoin["Close"], color='goldenrod', lw=2)
plt.plot(future_set["Date"], prediction, color='deeppink', lw=2)
plt.title("Bitcoin Price over time", size=25)
plt.xlabel("Time", size=20)
plt.ylabel("$ Price", size=20)
plt.show()
