import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, mean_scores, filename):
    fig=plt.figure()
 
    plt.plot(x,scores, label='Score')
    plt.plot(x,mean_scores, label='Rolling Avg 25 Obs.')
    plt.axhline(y=200, color="green", linestyle="--", label='Perfect Landing')
    
    plt.title('Learning Curve - DQN')
    plt.xlabel('Epoch')
    plt.ylabel('Score')

    plt.savefig(filename)

def error_plots(x,scores,sems,filename):
    plt.scatter(x,scores,label='Rolling Avg 25 Obs.')
    plt.xticks(x)
    plt.errorbar(x, scores, yerr=sems,fmt="o", label='SEM')
    plt.title('Average Score (Final 25 Obs) - DQN ')
    plt.xlabel('Monte Carlo')
    plt.ylabel('Score')
    plt.savefig(filename)