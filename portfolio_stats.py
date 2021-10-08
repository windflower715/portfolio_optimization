import pandas as pd 
import numpy as np 
import scipy
import scipy.optimize as sco 
import six
import scipy.cluster.hierarchy as sch
import cvxopt as opt
from cvxopt import blas, solvers
from finrl.marketdata.yahoodownloader import YahooDownloader
import plotly
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()
from pandas.tseries.offsets import *
import warnings
warnings.filterwarnings("ignore")

def get_price(assets,_start_date, _end_date):
    """returns two dataframes of price and daily returns
    assets: list of ticker strings,
    _start_date: string of start date ,
    _end_date: string of end date """
    df = YahooDownloader(start_date = _start_date,
                         end_date = _end_date,
                         ticker_list =assets ).fetch_data() #,'LQD','IEO','GLD'

    price = df.set_index(['date','tic'])[['close']].unstack()
    price.columns = price.columns.droplevel()
    price.index = pd.to_datetime(price.index)
    daily_rts = price/price.shift(1) -1
    price, daily_rts =price.resample('B').first().dropna(), daily_rts.resample('B').first().dropna()
    return price, daily_rts


class Portfolio():
    def __init__(self, daily_returns, daily_weights, rf ):
        self.daily_returns = daily_returns
        self.daily_weights = daily_weights
        self.rf = rf

    def portfolio_daily_returns(self): 
        """calculate portfolio daily returns
        returns a dataframe of daily returns"""
        pdaily_returns = pd.DataFrame( (self.daily_weights * self.daily_returns).sum(axis = 1), columns = ['portfolio_daily_returns']).dropna()
        return pdaily_returns

    def portfolio_mean_returns(self): 
        """calculate the total annualized average returns for the portfolio 
        over lifespan"""
        pmean_returns = (self.portfolio_daily_returns().mean()*260)[0]
        return pmean_returns 

    def portfolio_cum_returns(self): 
        """calculate portfolio cumulative returns
        returns a dataframe of cumulative returns"""
        pcum_returns =  (self.portfolio_daily_returns()+1).cumprod() -1
        pcum_returns.columns = ['portfolio_cum_returns']
        return pcum_returns

    def portfolio_total_returns(self): 
        """calculate portfolio end total returns
        returns a float of portfolio total returns"""
        ptotal_returns = self.portfolio_cum_returns()['portfolio_cum_returns'].dropna(how = 'any')[-1]
        return ptotal_returns

    def portfolio_sigma(self): 
        """calculate portfolio sigma/standard deviation
        returns a float of portfolio standard deviation"""
        pstdev = (self.portfolio_daily_returns().std() * np.sqrt(260))[0]
        return pstdev

    def portfolio_sharpe(self): 
        psharpe = (self.portfolio_mean_returns() - self.rf)/ self.portfolio_sigma()
        return psharpe    

    def portfolio_stats(self): 
        """returns a dataframe of portfolio total returns, mean returns, sigma and sharpe """
        pstats = pd.DataFrame(data = {'total_returns': self.portfolio_total_returns(),
        'mean_returns':[self.portfolio_mean_returns()], 'sigma':[self.portfolio_sigma()] ,
        'sharpe':[self.portfolio_sharpe()]})
        return pstats

    def portfolio_plot(self): 
        """plot cumulative returns of the portfolio"""
        cum_rts = self.portfolio_cum_returns()
        print(self.portfolio_stats())
        return cum_rts.iplot(title = 'Portfolio:' + ','.join(self.daily_returns) )

        
####portfolio optimization 

#define the objective function for scipy minimize-- here is the negative sharpe function
def neg_sharpe_ratio(weights,returns, risk_free_rate):
    p1 = Portfolio(returns, weights, risk_free_rate)
    return (-1)*p1.portfolio_sharpe()

def neg_total_returns(weights,returns, risk_free_rate):
    p1 = Portfolio(returns, weights, risk_free_rate)
    return (-1)*p1.portfolio_total_returns()

def neg_mean_returns(weights,returns, risk_free_rate):
    p1 = Portfolio(returns, weights, risk_free_rate)
    return (-1)*p1.portfolio_mean_returns()

def _sigma(weights,returns, risk_free_rate):
    p1 = Portfolio(returns, weights, risk_free_rate)
    return p1.portfolio_sigma()

##hierarchical risk parity functions
def getIVP(cov, **kargs):
    # Compute the inverse-variance portfolio
    ivp = 1. / np.diag(cov)
    ivp /= ivp.sum()
    return ivp


def getClusterVar(cov,cItems):
    # Compute variance per cluster
    cov_=cov.loc[cItems,cItems] # matrix slice
    w_=getIVP(cov_).reshape(-1,1)
    cVar=np.dot(np.dot(w_.T,cov_),w_)[0,0]
    return cVar


def getQuasiDiag(link):
    # Sort clustered items by distance
    link = link.astype(int)
    sortIx = pd.Series([link[-1, 0], link[-1, 1]])
    numItems = link[-1, 3]  # number of original items
    while sortIx.max() >= numItems:
        sortIx.index = range(0, sortIx.shape[0] * 2, 2)  # make space
        df0 = sortIx[sortIx >= numItems]  # find clusters
        i = df0.index
        j = df0.values - numItems
        sortIx[i] = link[j, 0]  # item 1
        df0 = pd.Series(link[j, 1], index=i + 1)
        sortIx = sortIx.append(df0)  # item 2
        sortIx = sortIx.sort_index()  # re-sort
        sortIx.index = range(sortIx.shape[0])  # re-index
    return sortIx.tolist()


def getRecBipart(cov, sortIx):
    # Compute HRP alloc
    w = pd.Series(1, index=sortIx)
    cItems = [sortIx]  # initialize all items in one cluster
    while len(cItems) > 0:
        cItems = [i[j:k] for i in cItems for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]  # bi-section
        for i in range(0, len(cItems), 2):  # parse in pairs
            cItems0 = cItems[i]  # cluster 1
            cItems1 = cItems[i + 1]  # cluster 2
            cVar0 = getClusterVar(cov, cItems0)
            cVar1 = getClusterVar(cov, cItems1)
            alpha = 1 - cVar0 / (cVar0 + cVar1)
            w[cItems0] *= alpha  # weight 1
            w[cItems1] *= 1 - alpha  # weight 2
    return w


def correlDist(corr):
    # A distance matrix based on correlation, where 0<=d[i,j]<=1
    # This is a proper distance metric
    dist = ((1 - corr) / 2.)**.5  # distance matrix
    return dist



class portfolio_optimizer():
    def __init__(self, daily_returns, rf ):

        self.daily_returns = daily_returns
        self.rf = rf
        self.cov = self.daily_returns.cov()
        self.corr = self.daily_returns.corr()

    def random_weights(self): 
        random_weights = pd.DataFrame( 
            columns = self.daily_returns.columns, 
        index = self.daily_returns.index)
        n_assets = len(self.daily_returns.columns)
        #dirichlet distribution generate n numbers that adds up to certain number
        weights = list(((np.random.dirichlet(np.ones(n_assets),size=1))).flatten())
        for idx, i  in enumerate(list(self.daily_returns.columns)): 
            random_weights[i] = weights[idx]
        return random_weights


    def mpt_weights(self, objective_function):
        """This funtion calculats static optimal weights according to modern porfolio
        theory using scipy optimize
        returns: a dataframe of daily returns
        rf: scalar value of risk free rate
        returns: a df of optimized weights based on MPT
        objective_function: str: 'max_sharpe', 'min_volatility', 
        'max_total_returns', 'max_mean_returns' """
        n_assets = len(self.daily_returns.columns)
        args = (self.daily_returns, self.rf)
        bounds = tuple((0,1) for asset in range(n_assets))
        if objective_function == 'max_sharpe': 
            result = sco.minimize(  neg_sharpe_ratio,
                            #initialize random weights for your portfolio
                            n_assets*[1./n_assets,], 
                            #args are a tuple of varaibles in your objective function that are given
                            args=args,
                            method='SLSQP', 
                            # another limit/bound to assign random weights
                            bounds=bounds, 
                            #constraint -- sum of x == 1(sum of all weights add up to 1)
                            constraints=({'type':'eq',
                            'fun':lambda x: np.sum (x)-1}))
        
        if  objective_function == 'min_volatility': 
            result = sco.minimize(  _sigma ,n_assets*[1./n_assets,], args=args, method='SLSQP', bounds=bounds, constraints=({'type':'eq','fun':lambda x: np.sum (x)-1}))
        
        if  objective_function == 'max_total_returns': 
            result = sco.minimize(  neg_total_returns ,n_assets*[1./n_assets,], args=args, method='SLSQP', bounds=bounds, constraints=({'type':'eq','fun':lambda x: np.sum (x)-1}))
        
        if  objective_function == 'max_mean_returns': 
            result = sco.minimize(  _sigma ,n_assets*[1./n_assets,], args=args, method='SLSQP', bounds=bounds, constraints=({'type':'eq','fun':lambda x: np.sum (x)-1}))

        mpt_weights = round(pd.DataFrame(result['x'], index = self.daily_returns.columns, columns = [self.daily_returns.index[0]] ) .T,2)
        mpt_weights = mpt_weights.reindex(self.daily_returns.index).fillna(method = 'ffill')

        return mpt_weights


    def hrp_weights(self): 
        # Construct a hierarchical portfolio
        dist = correlDist(self.corr)
        link = sch.linkage(dist, 'single')
        #dn = sch.dendrogram(link, labels=cov.index.values, label_rotation=90)
        #plt.show()
        sortIx = getQuasiDiag(link)
        sortIx = self.corr.index[sortIx].tolist()
        hrp = getRecBipart(self.cov, sortIx).sort_index()
        hrp = pd.DataFrame(dict(hrp), index = self.daily_returns.index)
        return hrp 


    def hrp_weights_dynamic(self, training_window,allocation_window, **kwargs ): 
        all_weights = pd.DataFrame()
        #loop the dates based on allocation window
        for i in range( training_window, len(self.daily_returns), allocation_window):
            training_df = self.daily_returns.iloc[i - training_window: i ].fillna(0.0001*2)

            opt = portfolio_optimizer( training_df, 0.02)
            hrp = opt.hrp_weights()
            hrpdic = dict(zip(hrp.columns, hrp.iloc[-1]))
            weightdf = pd.DataFrame(data = hrpdic, 
            index =pd.bdate_range(start =self.daily_returns.index[i] ,
             end = self.daily_returns.index[i] + BDay(allocation_window)) )

            all_weights = all_weights.append( weightdf)
        all_weights.index.name = 'asofdate'
        all_weights = all_weights.reset_index().drop_duplicates().set_index('asofdate')
        return all_weights


    def mpt_weights_dynamic(self,training_window,allocation_window,objective_function,**kwargs): 
        all_weights = pd.DataFrame()
        #loop the dates based on allocation window
        for i in range( training_window, len(self.daily_returns), allocation_window):
            training_df = self.daily_returns.iloc[i - training_window: i ].fillna(0.0001*2)

            opt = portfolio_optimizer( training_df, 0.02)
            mpt = opt.mpt_weights(objective_function)
            mptdic = dict(zip(mpt.columns, mpt.iloc[-1]))
            weightdf = pd.DataFrame(data = mptdic, 
            index =pd.bdate_range(start =self.daily_returns.index[i] ,
             end = self.daily_returns.index[i] + BDay(allocation_window)) )

            all_weights = all_weights.append( weightdf)
        all_weights.index.name = 'asofdate'
        all_weights = all_weights.reset_index().drop_duplicates().set_index('asofdate')
        return all_weights

class search_allocation():
    def __init__(self, assets, windows): 
        """assets: list of lists -[ ['SPY','QQQ','USO'],['SPY','USIG']]
           windows: list of lists -[[400,400],[360, 720]]"""
        self.assets = assets
        self.windows = windows

    def get_price_batch(self):
        # flatten list of tickers
        self.price, self.daily_rts = get_price(list(set(sum(self.assets,[]))), '1990-01-01','2050-01-01')

    def generate_weights(self,optimization_method, objective_function= None):
        """"This function generate weights given assets lists and window lists
            assets: list of lists of asset combos, [ ['SPY','QQQ','USO']], 
            windows: list of lists of training and allocation windows, [[150,150]]
            optimization_method: str, 'HRP' or 'MPT'
            objective_function: str, 'max_sharpe', 'min_volatility', 
            'max_total_returns', 'max_mean_returns'
            returns dynmic weights dictionary"""
        self.get_price_batch()
        self.weights = {}
        for i in self.assets:  
            daily_rts_subdf = self.daily_rts[i]
            optimizer = portfolio_optimizer(daily_rts_subdf, 0.02)
            for j in self.windows:             
                if optimization_method == 'HRP':
                    key  = '-'.join(i)  + ';' + optimization_method + ';' + ';'.join([str(x) for x in j]) 
                    self.weights[key] = optimizer.hrp_weights_dynamic(j[0], j[1])
                elif optimization_method == 'MPT':
                    key  = '-'.join(i) + ';' + optimization_method + ';' + objective_function + ';' + ';'.join([str(x) for x in j])
                    self.weights[key] = optimizer.mpt_weights_dynamic( j[0], j[1],objective_function )
        return self.weights

    def generate_portfolios(self, weights):
        """generate portfolios and hold it portfolios dictionary
        returns: a dictionary of portfolios objects with daily returns and weights
        weights: dictionary"""
        self.portfolios = {}
        for i in weights.keys():
            self.portfolios[i] = Portfolio(daily_returns= self.daily_rts[i.split(';')[0].split('-')], 
            daily_weights = weights[i], rf =  0.02)
        return self.portfolios

    def generate_portfolio_summary(self): 
        """"generate portfolio summary stats with different portfolios
        returns a dataframe"""
        portfolio_summary = pd.DataFrame(columns = ['total_returns', 'mean_returns', 'sigma', 'sharpe', 'assets',
            'opt_method', 'obj_fun', 'train_w', 'allo_w'] )
        for i in self.portfolios.keys(): 
            subdf = self.portfolios[i].portfolio_stats()
            #use generator method to assign columns
            try: 
                subdf['assets'], subdf['opt_method'], subdf['obj_fun'],subdf['train_w'], subdf['allo_w'] = (i.split(';')[j] for j in range(5) )
            except: 
                subdf['assets'], subdf['opt_method'],subdf['train_w'], subdf['allo_w'] = (i.split(';')[j] for j in range(4) )

            portfolio_summary = portfolio_summary.append(subdf)
            portfolio_summary = portfolio_summary.sort_values(by = 'total_returns')
        return portfolio_summary

    def portfolio_plots(self): 
        self.cum_returns = {}
        for j, i  in enumerate(self.portfolios.keys()):
             self.cum_returns[i] = self.portfolios[i].portfolio_cum_returns()
             self.cum_returns[i].columns = [i]
        cum_returns = pd.concat([self.cum_returns[i] for i in self.cum_returns.keys()])
        fig = cum_returns.iplot()

        return cum_returns 



def search_best_window(assets, windows, optimization_method, objective_function= None): 
    """This function searches for best assets and allocation window based on certain 
    optimization method and objective function:
    assets: list of lists of asset combos, [ ['SPY','QQQ','USO']], 
    windows: list of lists of training and allocation windows, [[150,150]]
    optimization_method: str, 'HRP' or 'MPT'
    objective_function: str, 'max_sharpe', 'min_volatility', 
        'max_total_returns', 'max_mean_returns'
    returns dynmic weights dictionary and performance dataframe """
    performance, weights = pd.DataFrame(), {}
    for i in assets: 
        price, daily_rts = get_price(i, '1990-01-01','2030-01-01')
        optimizer = portfolio_optimizer(daily_rts, 0.02)
        for j in windows: 
            key  = '-'.join(i) + '-' + optimization_method + '-' + objective_function + '-' + '-'.join([str(x) for x in j])
            if optimization_method == 'HRP':
                weights[key] = optimizer.hrp_weights_dynamic(j[0], j[1])
            elif optimization_method == 'MPT':
                weights[key] = optimizer.mpt_weights_dynamic( j[0], j[1],objective_function )

            #portfolio_df = pd.DataFrame(ps.portfolio(daily_rts,weights[key], 0.02).portfolio_stats(), index = [key]  )
            portfolio_df = (Portfolio(daily_rts,weights[key], 0.02).portfolio_stats() )
            portfolio_df.index = [key]
            performance = performance.append(portfolio_df)
    return weights, performance








if __name__ == '__main__': 
    print('portfolio_stats is being ran directly')
else: 
    pass



# price, daily_rts = get_price(['SPY','QQQ'])
# price, daily_rts =price.resample('B').first(), daily_rts.resample('B').first()
# daily_weights = pd.DataFrame( index = daily_rts.index, data = {'SPY':0.4,'QQQ':0.6}) 
# p1 = portfolio(daily_rts,daily_weights,0.02 )
# test  = p1.portfolio_plot()
# print(test)

