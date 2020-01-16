import urllib.request
from numpy import genfromtxt, zeros, mean, linspace, matrix, arange
import shutil
import tempfile
from pylab import plot, show, figure, subplot, hist, xlim
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score, homogeneity_score
from numpy.random import rand
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from numpy import corrcoef
from pylab import pcolor, colorbar, xticks, yticks
from sklearn.decomposition import PCA
from networkx import nx

# Data importing and Visualization

with urllib.request.urlopen('http://aima.cs.berkeley.edu/data/iris.csv') as response:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        shutil.copyfileobj(response, tmp_file)

with open(tmp_file.name) as html:
    data = genfromtxt(html, delimiter=',', usecols=(0, 1, 2, 3))

with open(tmp_file.name) as html:
    target = genfromtxt(html, delimiter=',', usecols=4, dtype=str)

plot(data[target == 'setosa', 0], data[target == 'setosa', 2], 'bo')
plot(data[target == 'versicolor', 0], data[target == 'versicolor', 2], 'ro')
plot(data[target == 'virginica', 0], data[target == 'virginica', 2], 'go')

show()

xmin = min(data[:, 0])
xmax = max(data[:, 0])
figure()
subplot(411)  # distribution of the setosa class (1st, on the top)
hist(data[target == 'setosa', 0], color='b', alpha=.7)
xlim(xmin, xmax)
subplot(412)  # distribution of the versicolor class (2nd)
hist(data[target == 'versicolor', 0], color='r', alpha=.7)
xlim(xmin, xmax)
subplot(413)  # distribution of the virginica class (3rd)
hist(data[target == 'virginica', 0], color='g', alpha=.7)
xlim(xmin, xmax)
subplot(414)  # global histogram (4th, on the bottom)
hist(data[:, 0], color='y', alpha=.7)
xlim(xmin, xmax)

show()

# Classification

t = zeros(len(target))
t[target == 'setosa'] = 1
t[target == 'versicolor'] = 2
t[target == 'virginica'] = 3

# Gaussian Naive Bayes Classification

classifier = GaussianNB()
classifier.fit(data, t)  # training on the iris dataset
train, test, t_train, t_test = train_test_split(data, t, test_size=0.4, random_state=0)
classifier.fit(train, t_train)  # train
print(classifier.score(test, t_test))  # test

# Confusion Matrix

print(confusion_matrix(classifier.predict(test), t_test))

# Classification Report

print(classification_report(classifier.predict(test), t_test, target_names=['setosa', 'versicolor', 'virginica']))

# cross validation with 6 iterations

scores = cross_val_score(classifier, data, t, cv=6)
print(scores)

# Mean accuracy

print(mean(scores))

# Kmeans Clustering

kmeans = KMeans(n_clusters=3, init='random')  # initialization
kmeans.fit(data)  # actual execution
c = kmeans.predict(data)
print(completeness_score(t, c))
print(homogeneity_score(t, c))
figure()
subplot(211)  # top figure with the real classes
plot(data[t == 1, 0], data[t == 1, 2], 'bo')
plot(data[t == 2, 0], data[t == 2, 2], 'ro')
plot(data[t == 3, 0], data[t == 3, 2], 'go')
subplot(212)  # bottom figure with classes assigned automatically
plot(data[c == 1, 0], data[c == 1, 2], 'bo', alpha=.7)
plot(data[c == 2, 0], data[c == 2, 2], 'go', alpha=.7)
plot(data[c == 0, 0], data[c == 0, 2], 'mo', alpha=.7)

show()

# Regression

x = rand(40, 1)  # explanatory variable
y = x * x * x + rand(40, 1) / 5  # dependent variable
linreg = LinearRegression()
linreg.fit(x, y)
xx = linspace(0, 1, 40)
plot(x, y, 'o', xx, linreg.predict(matrix(xx).T), '--r')

show()

# Mean squared_error

print(mean_squared_error(linreg.predict(x), y))

# Correlation

corr = corrcoef(data.T)  # .T gives the transpose
print(corr)
pcolor(corr)
colorbar()  # add
# arranging the names of the variables on the axis
xticks(arange(0.5, 4.5), ['sepal length', 'sepal width', 'petal length', 'petal width'], rotation=-20)
yticks(arange(0.5, 4.5), ['sepal length', 'sepal width', 'petal length', 'petal width'], rotation=-20)

show()

# Dimensionality Reduction

pca = PCA(n_components=2)
pcad = pca.fit_transform(data)
plot(pcad[target == 'setosa', 0], pcad[target == 'setosa', 1], 'bo')
plot(pcad[target == 'versicolor', 0], pcad[target == 'versicolor', 1], 'ro')
plot(pcad[target == 'virginica', 0], pcad[target == 'virginica', 1], 'go')

show()

print(pca.explained_variance_ratio_)
print(1 - sum(pca.explained_variance_ratio_))
data_inv = pca.inverse_transform(pcad)
print(abs(sum(sum(data - data_inv))))

for i in range(1, 5):
    pca = PCA(n_components=i)
    pca.fit(data)
    print(sum(pca.explained_variance_ratio_) * 100, '%')

# Network Mining
G = nx.read_gml('lesmiserables.gml')
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size=0, edge_color='b', alpha=.2)
nx.draw_networkx_labels(G, pos, edge_color='b', alpha=.2, font_size=10)
show()

'''
Gt = G.copy()

dn = nx.degree(Gt)

for n in Gt.nodes():
    if dn[n] <= 10:
        Gt.remove_node(n)

pos = nx.spring_layout(Gt)
nx.draw(Gt, pos, node_size=0, edge_color='b', alpha=.2)
nx.draw_networkx_labels(Gt, pos, edge_color='b', alpha=.2, font_size=10)
show()
'''