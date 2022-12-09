logs_root='logs'

import os
import sys
import subprocess
import jax.numpy as jnp
import matplotlib.pyplot as plt
from numpy import load
from vmcnet.utils import io
from vmcnet.models.construct import get_model_from_config
from vmcnet.postprocess.re_construct import get_model_from_config_nonsym
from vmcnet.postprocess.test_nonsym import assert_f_Af_slog as verify
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
    checkpointpaths=listpaths(os.path.join(path,'checkpoints'))
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

    return slog_psi,slog_f,paramshist,datahist,config

def slog_to_rel_ent(f,g):
    return jnp.average(f[1])-jnp.average(g[1])

def showfile(path):
    try: subprocess.Popen(['open',path])
    except: pass
    try: subprocess.Popen(['xdg-open',path])
    except: pass
    try: os.startfile(path)
    except: pass

if __name__=='__main__':

    path=pickrun(rootpath=logs_root)
    slog_Af,slog_f,paramshist,datahist,config=loadrun(path)

    AfX=[]
    fX=[]

    samplescutoff=250

    for i,(params,data) in enumerate(zip(paramshist,datahist)):

        if 'v' in sys.argv and i==0:
            n1,n2=config['problem']['nelec']
            print('\nverifying correct nonsym-antisym relation')
            verify(slog_f.apply,slog_Af.apply,params,data[0,:2,:,:],n1,n2,full=config['model']['full_det'])

        data=data[:,:samplescutoff,:,:]
        AfX.append(slog_Af.apply(params,data))
        fX.append(slog_f.apply(params,data))

        print('checkpoint {}/{}'.format(i+1,len(paramshist)))


    os.makedirs(os.path.join(path,'postprocessed'), exist_ok=True)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,5))

    ax1.plot([slog_to_rel_ent(f,Af) for f,Af in zip(fX,AfX)],'bo-',label='E[log(|f|/|Af|)]')
    ax1.legend()

    ax2.plot([jnp.average(f[-1]**2/Af[-1]**2) for f,Af in zip(fX,AfX)],'bo-',label='|f|^2/|Af|^2')
    ax2.legend()
    ax2.set_yscale('log')

    plotpath=os.path.join(path,'postprocessed/rel_ent.pdf') 
    plt.savefig(plotpath)

    outdatapath=os.path.join(path,'postprocessed/outdata')
    with open(outdatapath,'wb') as handle:
        pickle.dump({'f':fX,'Af':AfX},handle)

    showfile(os.path.split(plotpath)[0])


