import matplotlib.pyplot as plt
import seaborn as sns

def histvstarget(attribute, target, data):
    data[attribute].astype(float)
    total = data[[attribute, target]].groupby([attribute],as_index=False, sort=True).count()
    fig, ax1 = plt.subplots(1,1,figsize=(16,5))
    sns.barplot(x=attribute, y=target, data=total, ax=ax1)
    ax1.set(xlabel=attribute, ylabel=target)

def distribution(attribute, data):
    att = data[attribute]
    sns.distplot(att)