from mlxtend.frequent_patterns import fpgrowth
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd


def main():
    dataset =[
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
    print(frequent_itemsets)


if __name__ == '__main__':
    main()
