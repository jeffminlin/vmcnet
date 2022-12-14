logs_root='logs'

import os
import sys
import subprocess
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpy import load
from vmcnet.utils import io
from vmcnet.utils.slog_helpers import array_from_slog as del_slog
from vmcnet.models.construct import get_model_from_config
from vmcnet.postprocess.re_construct import get_model_from_config_nonsym
from vmcnet.postprocess.test_nonsym import assert_f_Af_slog as verify, noslog
import pickle
from jax import tree_util as tu, random as rnd
import math

def listpaths(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

def pickfrom(alloptions,allinfos):
    blocksize=9
    s=0
    while True:
        options,infos=alloptions[s:s+blocksize],allinfos[s:s+blocksize]
        optionstext='\n'.join(['{}: {} ({})'.format(i+1,p,info) for i,(p,info) in enumerate(zip(options,infos))])
        optionstext='u: up\n'+optionstext+'\nd: down'
        inp=input('\nInput number 1-{} or d=down/u=up (and press enter):\n{}\n'.format(len(options),optionstext))
        if inp=='d': s=(s+blocksize)%len(alloptions)
        if inp=='u': s=(s-blocksize)%len(alloptions)
        try:
            optionid=int(inp)-1
            return options[optionid]
        except:
            pass

def pickrun(rootpath):
    paths=list(reversed(sorted(listpaths(rootpath))))
    def getinfo(path):
        try:
            #return str(io.load_config_dict(path,'config.json')['problem'])[:100]
            return '{} checkpoints'.format(len(os.listdir(os.path.join(path,'checkpoints'))))
        except:
            return ''
    infos=[getinfo(p) for p in paths]
    return pickfrom(paths,infos)

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

def showfile(path):
    try: subprocess.Popen(['open',path])
    except: pass
    try: subprocess.Popen(['xdg-open',path])
    except: pass
    try: os.startfile(path)
    except: pass

if __name__=='__main__':

    try:
        path='.' if 'energy.txt' in os.listdir() else pickrun(rootpath=sys.argv[1])
    except:
        print('\nPass path to logs folder as first argument, e.g.,\npython postprocess.py logs\nor\npython postprocess.py .\n')
        quit()

    os.makedirs(os.path.join(path,'postprocessed'), exist_ok=True)
    mode='justplot' if 'justplot' in sys.argv else 'from_raw'
    plotdatapath=os.path.join(path,'postprocessed/plotdata')
    outdatapath=os.path.join(path,'postprocessed/outdata')

    if mode=='from_raw':
        slog_Af,slog_f,paramshist,datahist,timehist,config=loadrun(path)
        Af,f=noslog(slog_Af.apply),noslog(slog_f.apply)

        sl_AfX=[]
        sl_fX=[]
        PX=[]
        PX_now=[]
        M=10

        for i,(params,data) in enumerate(zip(paramshist,datahist)):
            if 'staticsamples' in sys.argv:
                print('using static samples')
                sigma1=jnp.sqrt(jnp.average(datahist[0]**2))
                sigma2=jnp.sqrt(jnp.average(datahist[-1]**2))
                sigma=max(sigma1,sigma2)
                if i==0:
                    X=sigma*rnd.normal(rnd.PRNGKey(0),data.shape)
                    p=jnp.exp(-jnp.sum(X**2,axis=(-2,-1))/(2*sigma))
                    p=p/jnp.sum(p)
            else:
                if i%M==0:
                    X=data
                    p=Af(params,X)**2
                    p=p/jnp.sum(p)

            if 'v' in sys.argv and i==0:
                n1,n2=config['problem']['nelec']
                print('\nverifying correct nonsym-antisym relation')
                verify(slog_f.apply,slog_Af.apply,params,X[0,:2,:,:],n1,n2,full=config['model']['full_det'])

            sl_AfX.append(slog_Af.apply(params,X))
            sl_fX.append(slog_f.apply(params,X))

            PX.append(p)
            p_now=del_slog(sl_AfX[-1])**2
            p_now=p_now/jnp.sum(p_now)
            PX_now.append(p_now)

            print('checkpoint {}/{}'.format(i+1,len(paramshist)))


        with open(outdatapath,'wb') as handle:
            pickle.dump({'timehist':timehist,\
                'sl_fX':sl_fX,'sl_AfX':sl_AfX,'PX':PX,'PX_now':PX_now,\
                'paramshist':paramshist},handle)

    if mode=='justplot':
        with open(outdatapath,'rb') as handle:
            outdata=pickle.load(handle)
        for k,v in outdata.items():
            globals()[k]=v

    unweight=[1/p for p in PX]
    fX=[del_slog(sl) for sl in sl_fX]
    AfX=[del_slog(sl) for sl in sl_AfX]
    fnorms=[jnp.sum(uw*p*f**2) for f,p,uw in zip(fX,PX_now,unweight)]
    Afnorms=[jnp.sum(uw*p*Af**2) for Af,p,uw in zip(AfX,PX_now,unweight)]
    p1=[jnp.average(slf[1]/slAf[1]) for slf,slAf in zip(sl_fX,sl_AfX)]

    WLnorms=[[jnp.average(L**2) for L in tu.tree_leaves(params)] for params in paramshist]
    #Wnorms=[functools.reduce(lambda x,y:x+y,wln) for wln in WLnorms]
    Wnorms=[jnp.max(jnp.array(wln)) for wln in WLnorms]

    #p1=[jnp.sum(uw*p_now*(slf[1]/slAf[1])) for slf,slAf,p_now,uw in zip(sl_fX,sl_AfX,PX_now,unweight)]
    #fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))
    #ax1.plot(timehist,p1,'bo-',label='E[log(|f|/|Af|)]')
    #ax1.legend()

    fig,(ax1,ax2,ax3)=plt.subplots(3,1,figsize=(7,14))
    #ax1.plot(timehist,p1,'bo-',label='E[log(|f|/|Af|)]')

    ax1.plot(timehist,[f/Af for f,Af in zip(fnorms,Afnorms)],'bo-',label='|f|^2/|Af|^2')
    ax1.legend()
    ax1.set_yscale('log')
    ax2.plot(timehist,fnorms,'bo-',label='|f|^2')
    ax2.plot(timehist,Wnorms,'sr:',label='|W|')
    ax2.legend()
    ax2.set_yscale('log')
    ax3.plot(timehist,Afnorms,'bo-',label='|Af|^2')
    ax3.legend()
    ax3.set_yscale('log')

    plotpath=os.path.join(path,'postprocessed/rel_ent.pdf') 
    plt.savefig(plotpath)

    opath=os.path.split(plotpath)[0]
    print(opath)
    showfile(opath)


