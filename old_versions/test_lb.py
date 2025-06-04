### test lower bound of pure exploration


from kjunutils3_v2 import *
import cvxpy as cp



def dGaussian(p,q,sigma_sq):
    return (p-q)**2/(2*sigma_sq);

def dBernoulli(p,q):
    eps = 1e-16
    res=0
    if (p!=q):
        if (p<=0):
            p = eps
        if (p>=1):
            p = 1-eps
        res=(p*log(p/q) + (1-p)*log((1-p)/(1-q))) 
    return res

#div = dBernoulli
#div = dGaussian

def binSearch(f, xMin, xMax, tol=1e-11):
    # find m such that f(m)=0 using dichotomix search
    l = xMin
    u = xMax
    sgn = f(xMin)
    while u-l>tol:
        m = (u+l)/2
        if f(m)*sgn>0:
            l = m
        else:
            u = m
    m = (u+l)/2
    return m

def I(alpha,mu1,mu2,div):
    if (alpha==0) or (alpha==1):
        return 0
    else:
        mid=alpha*mu1 + (1-alpha)*mu2
        return alpha*div(mu1,mid)+(1-alpha)*div(mu2,mid)

def muddle(mu1, mu2, nu1, nu2):
    return (nu1*mu1 + nu2*mu2)/(nu1+nu2)

def cost(mu1, mu2, nu1, nu2,div):
    if (nu1==0) and (nu2==0):
        return 0 
    else:
        alpha=nu1/(nu1+nu2)
        return((nu1 + nu2)*I(alpha,mu1,mu2,div))

def xkofy(y, k, mu, div, tol = 1e-11):
    # return x_k(y), i.e. finds x such that g_k(x)=y
    def g(x):
        return (1+x)*cost(mu[0], mu[k], 1/(1+x), x/(1+x), div)-y
    xMax=1
    while g(xMax)<0:
        xMax *= 2
    return binSearch(lambda x: g(x), 0, xMax, 1e-11)

def aux(y,mu,div):
    # returns F_mu(y) - 1
    K = len(mu)
    x = [xkofy(y, k, mu, div) for k in range(1,len(mu))]
    m = [muddle(mu[0], mu[k], 1, x[k-1]) for k in range(1,len(mu))]
    return (sum([div(mu[0],m[k-1])/(div(mu[k], m[k-1])) for k in range(1,len(mu))])-1.0)

def oneStepOpt(mu, div=dBernoulli, tol=1e-11):
    objMax=0.5
    if div(mu[0], mu[1])==Inf:
        # find objMax such that aux(objMax,mu)>0
        while aux(objMax,mu,div)<0:
           objMax=objMax*2
    else:
        objMax=div(mu[0],mu[1])

#    printExpr("objMax")
    y = binSearch(lambda y: aux(y,mu,div), 0, objMax, tol)
#    printExpr("y")
    beta = [xkofy(y, k, mu, div, tol) for k in range(1,len(mu))]
    beta = np.concatenate( ([1.0], beta) )
    alpha = beta/sum(beta)
    return alpha[0]*y, alpha
    # y is 
    # nu is the alpha.. the ratio.

def OptimalWeights(mu, div=dBernoulli, tol=1e-11):
    # returns T*(mu) and w*(mu)
    # T^*(mu): the constant
    # w^*(mu): the ratio
    K=len(mu) # of arms
    IndMax=np.where(mu==max(mu))[0] #    IndMax=findall(mu.==maximum(mu))
    nTies=len(IndMax) # num of ties
    if (nTies>1):
         # multiple optimal arms
         nuOpt=zeros(K)
         nuOpt[IndMax]=1.0/nTies
         return 0,nuOpt
    else:
         mu=mu.copy()    # make a local copy
         index = argsort(mu)[::-1]   # sort large to small
         mu=mu[index] 
         unsorted=np.arange(K) #vec(collect(1:K))
         invindex=zeros(K,dtype=int)
         invindex[index]=unsorted 
         # one-step optim
         [fmax,alpha_]=oneStepOpt(mu,div,tol) #mu is sorted one
         # back to good ordering
         alpha=alpha_[invindex]
         return fmax,alpha.copy()

def solve_inner(w, mu, div=dBernoulli):
    assert( all(diff(mu) <= 0))
    K = len(mu)
    lams = []
    objs = []
    for a in range(1,K):
        p = w[0] / (w[0] + w[a])
        lam = p*mu[0] + (1-p)*mu[a]
        obj = w[0]*div(mu[0],lam) + w[a]*div(mu[a],lam)
        lams.append( lam )
        objs.append( obj )
    i = np.argmin( objs )
    best_obj = objs[i]
    best_lam = lams[i]
    the_lam = mu.copy()
    the_lam[[0,i+1]] = best_lam
#    ipdb.set_trace()
    return the_lam
    
# def I(mu1,mu2,a,sig_sq=1.0):
#     mubar = a*mu1 + (1-a)*mu2
#     return a*dGaussian(mu1, mubar, sig_sq) + (1-a)*dGaussian(mu2, mubar, sig_sq)

def main():
    #dd = dBernoulli
    sig_sq = 1**2
    mydiv = lambda p,q: dGaussian(p,q,sig_sq)

    mu = np.array([0.3, 0.25, 0.2, 0.1])

    [fmax,alpha] = oneStepOpt(mu,mydiv)
    printExpr('(fmax,alpha)')
    tmp = mu-mu[0]
    tmp[1:] = 2/tmp[1:]**2
    tmp[0] = tmp[0]
    naive_const = 4*tmp.sum() # this seems correct, given the constant of 8 in (c) of sec 33.2.1 of bandit algorithms book

    ipdb.set_trace()

    optimal_const = 1.0/fmax
    printExpr('optimal_const')
    printExpr('naive_const')
    # TODO c(obj,w) = OptimalWeights(mu2,dd) ompute what is the ratio for the optimal_const!!

    sig_sq = 1**2
    dd = lambda p,q: dGaussian(p,q,sig_sq)
    x = alpha / alpha[0]
    m = (mu[0] + x*mu)/(1 + x)
    
    for i in range(1, len(mu)):
        printExpr('dd(mu[0],m[i]) + x[i]*dd(mu[i], m[i])')
#         val = (alpha[0] + alpha[i])*I(alpha[0]/(alpha[0]+alpha[i]), mu[0], mu[i], mydiv)
#         printExpr('val')

    for i in range(1, len(mu)):
        printExpr(' dd(mu[0],mu[i])/(1/alpha[0] + 1/alpha[i])')

    ipdb.set_trace()

    mu2 = np.array([0.25, 0.3, 0.2, 0.1])
    mu3 = np.array([0.40, 0.01])
#     printExpr('OptimalWeights(mu2,dd)')
#     printExpr('oneStepOpt(mu3,dd)')

    #--------------------
    sig_sq = 1**2
    dd = lambda p,q: dGaussian(p,q,sig_sq)
    # dd = dGaussian
    mu4 = np.array([0.3, 0.30, 0.1])
    printExpr('mu4')
    #w = np.ones(len(mu4))/len(mu4)
    (obj,w) = OptimalWeights(mu4,dd) 
    printExpr('w')
    lam = solve_inner(w, mu4, dd)
    printExpr('lam')
    ipdb.set_trace()

    # print('============= gaussian')
    # mu4 = np.array([0.5, 0.3, 0.2])

if __name__ == '__main__':
    main()

