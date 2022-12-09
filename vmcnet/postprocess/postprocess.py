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

    path=pickrun(rootpath=logs_root)
    slog_Af,slog_f,paramshist,datahist,timehist,config=loadrun(path)
    Af,f=noslog(slog_Af.apply),noslog(slog_f.apply)

    sl_AfX=[]
    sl_fX=[]
    PX=[]
    PX_now=[]
    M=10

    for i,(params,data) in enumerate(zip(paramshist,datahist)):
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

    unweight=[1/p for p in PX]

    os.makedirs(os.path.join(path,'postprocessed'), exist_ok=True)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))

    p1=[jnp.sum(uw*p_now*(slf[1]/slAf[1])) for slf,slAf,p_now,uw in zip(sl_fX,sl_AfX,PX_now,unweight)]
    ax1.plot(timehist,p1,'bo-',label='E[log(|f|/|Af|)]')
    ax1.legend()

    fX=[del_slog(sl) for sl in sl_fX]
    AfX=[del_slog(sl) for sl in sl_AfX]
    fnorms=[jnp.sum(uw*p*f**2) for f,p,uw in zip(fX,PX_now,unweight)]
    Afnorms=[jnp.sum(uw*p*Af**2) for Af,p,uw in zip(AfX,PX_now,unweight)]

    ax2.plot(timehist,[f/Af for f,Af in zip(fnorms,Afnorms)],'bo-',label='|f|^2/|Af|^2')
    ax2.legend()
    ax2.set_yscale('log')

    breakpoint()

    plotpath=os.path.join(path,'postprocessed/rel_ent.pdf') 
    plt.savefig(plotpath)

    outdatapath=os.path.join(path,'postprocessed/plotdata')
    with open(outdatapath,'wb') as handle:
        pickle.dump({'times':timehist,'fnorms':fnorms,'Afnorms':Afnorms,'rel_ent':p1},handle)
    if 'sd' in sys.argv:
        outdatapath=os.path.join(path,'postprocessed/outdata')
        with open(outdatapath,'wb') as handle:
            pickle.dump({'times':timehist,'f':fX,'Af':AfX,'P':PX},handle)

    showfile(os.path.split(plotpath)[0])


