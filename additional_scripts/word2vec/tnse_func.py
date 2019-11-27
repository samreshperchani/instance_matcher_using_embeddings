import matplotlib.pyplot as plt
import matplotlib.lines as mlines

def tsne_plot(model, word_1, word_2):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []
    count_of_neighbours = 5
    tokens.append(model.wv[word_1])
    labels.append(word_1.replace('http://dbkwik.webdatacommons.org/memory_alpha','memory-alpha'))

    tokens.append(model.wv[word_2])
    labels.append(word_2.replace('http://dbkwik.webdatacommons.org/memory_beta','memory-beta'))

    for neighbour in model.wv.most_similar(word_1, topn= count_of_neighbours):
        tokens.append(model.wv[neighbour[0]])
        labels.append(neighbour[0].replace('http://dbkwik.webdatacommons.org/memory_alpha','memory-alpha'))

    for neighbour in model.wv.most_similar(word_2, topn= count_of_neighbours):
        tokens.append(model.wv[neighbour[0]])
        labels.append(neighbour[0].replace('http://dbkwik.webdatacommons.org/memory_beta','memory-beta'))
    
    tsne_model = TSNE(perplexity=5, n_components=2, init='pca', n_iter=3000, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(10, 10)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i], c = 'g' if str(labels[i]).find('memory-alpha')!=-1 else 'b')
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha= 'left' if str(labels[i]).find('memory-alpha')!=-1 else 'right',
                     va='bottom')
    #plt.show()
    green_circle = mlines.Line2D([], [], color='green', marker='o', linestyle='None',
                          markersize=10, label='memory-alpha')
    blue_circle = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                          markersize=10, label='memory-beta')
                          
    plt.legend(handles=[green_circle, blue_circle])
    plt.rcParams.update({'font.size': 12})
    plt.savefig('plot.png')