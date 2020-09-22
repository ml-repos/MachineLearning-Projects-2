library(arules)

dataset = read.csv('market.csv',header = FALSE)
dataset = read.transactions('market.csv',sep = ',',rm.duplicates = TRUE)

summary(dataset)
itemFrequencyPlot(dataset,topN=10)

rules = apriori(data = dataset,parameter = list(support = 0.004,confidence = 0.2))
rules1 = eclat(data = dataset,parameter = list(support = 0.004,minlen = 2))

inspect(sort(rules,by = 'lift')[1:10])

inspect(sort(rules1,by = 'support')[1:10])
