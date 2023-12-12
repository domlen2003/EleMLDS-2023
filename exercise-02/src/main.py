from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np


def main():
    '''dataset =[
        ['A', 'B', 'C'],
        ['B', 'D'],
        ['A', 'C', 'E', 'F'],
        ['A', 'C', 'E', 'F', 'G'],
        ['B', 'C', 'E'],
        ['A', 'C', 'D', 'E', 'F'],
        ['A', 'B', 'C', 'G'],
        ['A', 'E', 'F', 'H'],
        ['C', 'D', 'E'],
        ['A', 'B', 'C', 'G', 'H']
    ]

    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    print(df)

    frequent_itemsets = fpgrowth(df, min_support=0.4, use_colnames=True)

    frequent_itemsets = frequent_itemsets.sort_values(by=['support'], ascending=False)
    print(frequent_itemsets)'''

    # Defining the data based on the provided split points
    # The format is (leaf1, leaf2, distance, number of items in cluster)
    linkage_matrix = [
        [1., 5., 2., 2.],  # t1
        [2., 4., 2., 2.],  # t2
        [7., 8., 2., 3.],  # t3, where 7 is t2's new index after adding t1
        [0., 3., 3., 3.],  # t4, where 0 is t1's index
        [6., 9., 3., 4.],  # t5, where 6 is t3's new index after adding t4
        [5., 10., 4., 7.],  # t6, where 5 is t4's new index and 10 is t5's
        [11., 7., 4., 8.],  # t7, where 11 is t6's new index
        [12., 9., 4., 9.]  # t8, where 12 is t7's new index
    ]

    # Convert to a numpy array
    linkage_matrix = np.array(linkage_matrix)

    # Create the dendrogram
    dendrogram(linkage_matrix)

    # Display the plot
    plt.title("Dendrogram")
    plt.xlabel("Index")
    plt.ylabel("Distance")
    plt.show()


if __name__ == '__main__':
    main()
