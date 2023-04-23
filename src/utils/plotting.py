import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, mean_scores, filename, plotType):

    if plotType == 'score':
        ylabel = 'Score'
        title = ylabel
    elif plotType == 'loss':
        ylabel = 'Loss'
        title = ylabel
    else:
        print("Invalid input for plot type!")

    plt.figure()
 
    plt.plot(x,scores, label=ylabel)
    plt.plot(x,mean_scores, label='Rolling Avg 25 Obs.')
    plt.axhline(y=200, color="green", linestyle="--", label='Perfect Landing')
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.legend()

    plt.savefig(filename)

def error_plots(x,scores,sems,filename):
    plt.figure()

    plt.scatter(x,scores,label='Rolling Avg 25 Obs.')
    plt.xticks(x)
    plt.errorbar(x, scores, yerr=sems,fmt="o", label='SEM')

    plt.title('Average Score (Final 25 Obs) - DQN ')
    plt.xlabel('Monte Carlo')
    plt.ylabel('Score')
    plt.legend()

    plt.savefig(filename)