# example:
# python PATH-TO-postprocess.py logpath=PATH-TO-logs
#
# or from logs dir OR logs/20220101-120000:
# python PATH-TO-postprocess.py

import os
import sys
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpy import load
from vmcnet.utils import io
from vmcnet.models.construct import get_model_from_config
from vmcnet.postprocess.re_construct import get_model_from_config_nonsym
from vmcnet.postprocess.test_nonsym import assert_f_Af_slog as verify, noslog
from vmcnet.postprocess.pickrun import listpaths,pickrun,showfile
import pickle
from jax import tree_util as tu, random as rnd
import math


def loadrun(path):
    gettime=lambda p: int(os.path.split(p)[-1][:-4])

    checkpointpaths=listpaths(os.path.join(path,'checkpoints'))
    checkpointpaths.sort(key=gettime)
    config=io.load_config_dict(path,'config.json')
    model=config['model']
    problem=config['problem']

    alldatahist=[load(p,allow_pickle=True) for p in checkpointpaths]
    paramshist=[ad['p'].tolist() for ad in alldatahist]
    datahist=[ad['d'].tolist()['walker_data']['position'] for ad in alldatahist]

    args=(model, problem['nelec'], jnp.array(problem['ion_pos']), jnp.array(problem['ion_charges']))
    kwargs=dict(dtype=config['dtype'])

    slog_psi = get_model_from_config(*args,**kwargs)
    print('Loaded model')
    slog_f = get_model_from_config_nonsym(*args,**kwargs)
    print('Loaded nonsymmetrized model')

    return slog_psi,slog_f,paramshist,datahist,[gettime(p) for p in checkpointpaths],config


if __name__=='__main__':

    samplesbound=10000      # run with arg samplesbound=100 to speed up
    mode='from_raw_data'    # run with arg mode=justplot to quickly modify the visuals of previously generated plots

    for s in sys.argv[1:]:
        if '=' in s:
            k,v=s.split('=')
            try: v=int(v)
            except: pass
            globals()[k]=v

    #################### options ####################
    #
    # __To process pre-selected run__:
    # run from 20220101-120000 or with arg logpath=20220101-120000
    #
    # __To browse logs directory and process selected run__:
    # run from logs directory or with arg logpath=PATH-TO-LOGS-DIR 
    #
    #
    #################################################

    try: path=pickrun(logpath)
    except: path=pickrun('.')

    os.makedirs(os.path.join(path,'postprocessed'), exist_ok=True)

    plotdatapath=os.path.join(path,'postprocessed/plotdata')
    outdatapath=os.path.join(path,'postprocessed/outdata')

    slog_Af,slog_f,paramshist,datahist,timehist,config=loadrun(path)
    if mode=='from_raw_data':
        Af,f=noslog(slog_Af.apply),noslog(slog_f.apply)

        sl_AfX=[]
        sl_fX=[]

        # print('using static samples')
        # _,_,n,d=datahist[0].shape
        # var1=jnp.average(datahist[0]**2)
        # var2=jnp.average(datahist[-1]**2)
        # sigma=jnp.sqrt(var1+var2)
        # X=sigma*rnd.normal(rnd.PRNGKey(0),(1,samples,n,d))
        # print(sigma)
        # p=jnp.exp(-jnp.sum(X**2,axis=(-2,-1))/(2*sigma**2))/(jnp.sqrt(2*math.pi)*sigma)**(n*d)
        # breakpoint()

        for i,(params,X) in enumerate(zip(paramshist,datahist)):
            X=X[:,:samplesbound,:,:]
            if 'v' in sys.argv and i==0:
                n1,n2=config['problem']['nelec']
                print('\nverifying correct nonsym-antisym relation')
                verify(slog_f.apply,slog_Af.apply,params,X[0,:2,:,:],n1,n2,full=config['model']['full_det'])

            sl_AfX.append(slog_Af.apply(params,X))
            sl_fX.append(slog_f.apply(params,X))
            print('checkpoint {}/{}'.format(i+1,len(paramshist)))

        with open(outdatapath,'wb') as handle:
            pickle.dump({'sl_fX':sl_fX,'sl_AfX':sl_AfX},handle)

    if mode=='justplot':
        with open(outdatapath,'rb') as handle:
            outdata=pickle.load(handle)
        for k,v in outdata.items():
            globals()[k]=v

    # fX=[array_from_slog(sl) for sl in sl_fX]
    # AfX=[array_from_slog(sl) for sl in sl_AfX]
    # fnorms=[jnp.average(f**2/p) for f,p in zip(fX,PX)]
    # Afnorms=[jnp.average(Af**2/p) for Af,p in zip(AfX,PX)]

    order_of_mag_f=[jnp.average(sl[1]) for sl in sl_fX]
    order_of_mag_Af=[jnp.average(sl[1]) for sl in sl_AfX]

    relative_order=[f-Af for f,Af in zip(order_of_mag_f,order_of_mag_Af)]

    WLnorms=[[jnp.average(L**2) for L in tu.tree_leaves(params)] for params in paramshist]
    Wnorms=[jnp.max(jnp.array(wln)) for wln in WLnorms]

    opath=os.path.join(path,'postprocessed')
    info={b:str(config[a][b]) for a,b in [('problem','nelec'),('model','ndeterminants')]}

    for suffix in ['','_xlog']:
    #for suffix in ['']:
        fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(7,14))
        fig.suptitle(' '.join(['{}={}'.format(a,b) for a,b in info.items()]))
        ax1.plot(timehist,relative_order,'b-',label='E[log(|f(X)|/|Af(X)|)]')
        ax2.plot(timehist,order_of_mag_f,'r--',label='E[log|f(X)|]')
        ax2.plot(timehist,order_of_mag_Af,'b-',label='E[log|Af(X)|]')
        ax3.plot(timehist,Wnorms,'r',label='|W|')
        for ax in (ax1,ax2,ax3):
            ax.legend()
            if 'xlog' in suffix:
                ax.set_xscale('log')
        plt.savefig(os.path.join(opath,'rel_ent{}.pdf'.format(suffix)))

    print('output path:')
    print(opath)
    showfile(opath)


