import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def f(x):
    x1,x2=x[0],x[1]
    return (1-x1)**2+100*(x2-x1**2)**2

def df(x):
    x1,x2=x[0],x[1]
    d1=2*(x1-1) + 400*(x1**2-x2)*x1
    d2=200*(x2-x1**2)
    v=np.array([d1,d2])
    return v, np.sqrt(np.sum(v**2)) 
    
def H(x):
    x1,x2=x[0],x[1]
    return np.array([[2+1200*x1**2-400*x2, -400*x1],[-400*x1, 200]])

def T(x):
    x1,x2=x[0],x[1]
    return np.array([[[2400*x1, -400],[-400, 0]],[[-400, 0],[0, 0]]])

def evaluation(x, count):
    count+=1
    return f(x), df(x), H(x), count
    
def plots(hist, itera, name, X, Y, partb=False):
    
    left  = 0.125  # the left side of the subplots of the figure
    right = 0.9    # the right side of the subplots of the figure
    bottom = 0.1   # the bottom of the subplots of the figure
    top = 0.9      # the top of the subplots of the figure
    wspace = 0.25   # the amount of width reserved for blank space between subplots
    hspace = 0.3   # the amount of height reserved for white space between subplots

    #Plot
    plt.figure(figsize=(10,3.5))
    plt.subplots_adjust(left, bottom, right, top, wspace, hspace)

    ### Plot 1###
    plt.subplot(1, 2, 1)
    plt.plot(np.log10(hist['ndf']), label='$log ||\\nabla f(x_t)||$')
    plt.plot(np.log10(hist['f']), label='$log f(x_t)$')
    plt.legend(bbox_to_anchor=(.975, .975), loc='upper right', borderaxespad=.0)
    
    if partb: 
        plt.xlabel('Function Evaluations', size=12)
    else:
        plt.xlabel('Descent Iterations', size=12)
        
    plt.ylabel('$log_{10}$', size=12)
    plt.title("$log_{10}$ of Gradient Norm and Objective Function")

    ### Plot 2###
    m=X.shape[0]
    plt.subplot(1, 2, 2)

    Z=np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            Z[i,j] = f(np.array([X[i,j],Y[i,j]]))

    contours=plt.contour(X, Y, Z, levels=np.logspace(-1,3,5), alpha=.5)
    plt.clabel(contours)

    for j in range(1,itera):
        xy=(hist['x'][j][0],hist['x'][j][1])
        xytext=(hist['x'][j-1][0],hist['x'][j-1][1])

        plt.annotate('', xy=xy, xytext=xytext,
                           arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                           va='center', ha='center')
        plt.xlim(np.min(X), np.max(X))
        plt.ylim(np.min(Y), np.max(Y))

    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))
    plt.xlabel('X1', size=12)
    plt.ylabel('X2', size=12)
    plt.title("Descent Iterations in Objetive Function Surface")
    
    plt.savefig('plots/'+name+'.png', bbox_inches='tight',dpi=300)
    plt.show()