from browse import listpaths,browse

######################################################################################################################

import os
import jax.numpy as jnp
from numpy import load
from vmcnet.utils import io
from vmcnet.models.construct import get_model_from_config


def loadrun(path):
    gettime=lambda p: int(os.path.split(p)[-1][:-4])

    checkpointpaths=listpaths(os.path.join(path,'checkpoints'))
    checkpointpaths.sort(key=gettime)
    config=io.load_config_dict(path,'config.json')
    model_config=config['model']
    problem=config['problem']

    alldatahist=[load(p,allow_pickle=True) for p in checkpointpaths]
    paramshist=[ad['p'].tolist() for ad in alldatahist]
    datahist=[ad['d'].tolist()['walker_data']['position'] for ad in alldatahist]

    args=(model_config, problem['nelec'], jnp.array(problem['ion_pos']), jnp.array(problem['ion_charges']))
    kwargs=dict(dtype=config['dtype'])

    model = get_model_from_config(*args,**kwargs)
    print('Loaded model')
    return model,paramshist,datahist,[gettime(p) for p in checkpointpaths],config



def loadsnapshot(path):
    model,paramshist,datahist,timehist,config=loadrun(path)
    params=paramshist[-1]
    X=datahist[-1]
    n,d=X.shape[-2:]
    if len(X.shape)==4:
        X=jnp.reshape(X,(-1,n,d))

    return model,params,X,config


if __name__=='__main__':
    logpath='logs'
    try: path=browse(logpath)
    except: path=browse('.')
    model,params,X=loadsnapshot(path)



#var_gaussian=jnp.average(datahist[0]**2+datahist[-1]**2)
#X_gaussian=rnd.normal(rnd.PRNGKey(0),(1,min(samples,samplesbound),n,d))*jnp.sqrt(var_gaussian)
#logp_gaussian=-jnp.sum(X_gaussian**2/(2*var_gaussian),axis=(-2,-1))-0.5*jnp.log(2*math.pi*var_gaussian)*n*d
