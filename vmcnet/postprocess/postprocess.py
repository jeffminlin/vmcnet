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

def listpaths(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

def pickfrom(options,infos):
    optionstext='\n'.join(['{}: {} ({})'.format(i+1,p,info) for i,(p,info) in enumerate(zip(options,infos))])
    inp=input('\nInput number 1-{} and press enter:\n{}\n'.format(len(options),optionstext))
    optionid=int(inp)-1
    return options[optionid]

def pickrun(cutoff=10):
    paths=list(reversed(sorted(listpaths('logs'))))
    paths=paths[:cutoff]
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
    slog_f = get_model_from_config_nonsym(*args,**kwargs)

    return slog_psi,slog_f,paramshist,datahist,config

def slog_to_rel_ent(fX,AfX):
    _,l=fX
    _,lA=AfX
    return jnp.average(l)-jnp.average(lA)

def showfile(path):
    try: subprocess.Popen(['open',path])
    except: pass
    try: subprocess.Popen(['xdg-open',path])
    except: pass
    try: os.startfile(path)
    except: pass

if __name__=='__main__':

    path=pickrun()
    slog_Af,slog_f,paramshist,datahist,config=loadrun(path)

    AfX=[]
    fX=[]

    print(config)

    for i,(params,data) in enumerate(zip(paramshist,datahist)):

        if 'v' in sys.argv and i==0:
            n1,n2=config['problem']['nelec']
            print('\nverifying correct nonsym-antisym relation')
            verify(slog_f.apply,slog_Af.apply,params,data[0,:5,:,:],n1,n2,full=config['model']['full_det'])

        AfX.append(slog_Af.apply(params,data))
        fX.append(slog_f.apply(params,data))

        print('checkpoint {}/{}'.format(i+1,len(paramshist)))

    plt.plot([slog_to_rel_ent(f,Af) for f,Af in zip(fX,AfX)],'bo-',label='E[log(|f|/|Af|)]')
    plt.legend()

    os.makedirs(os.path.join(path,'postprocessed'), exist_ok=True)
    plotpath=os.path.join(path,'postprocessed/ratio.pdf') 

    plt.savefig(plotpath)
    showfile(plotpath)


