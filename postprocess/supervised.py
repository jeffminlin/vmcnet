import jax.numpy as jnp
import jax.random as rnd
import sys
import optax
from flax.training.train_state import TrainState
from mcmc.session import nextkey
import jax
import mcmc.tests
import mcmc.energy
import mcmc.mcmc
import load,browse
from tqdm import tqdm
from vmcnet.physics.core import initialize_molecular_pos

if 'dj' in sys.argv:
    jax.config.update('jax_disable_jit', True)
    print('disabled jit')


if 'm' in sys.argv:
    mode='mcmc'
elif 'c' in sys.argv:
    mode='correlated'
elif 'n' in sys.argv:
    mode='no_mc'
else:
    print('\nchoose mode m=MCMC, c=correlated, n=NO_MC\n')
    quit()

logpath='../logs'
try: path=browse.browse(logpath)
except: path=browse.browse('.')
fi_model,fi_params,Y,config=load.loadsnapshot(path)
si_model=fi_model

# set learner to the same model
slog_Si=si_model.apply
Si=mcmc.energy.noslog(slog_Si)
P_Si=lambda P,X: Si(P,X)**2

params0=si_model.init(nextkey(),jnp.ones_like(Y[0:2]))

@jax.jit
def fi(X):
    s,l=fi_model.apply(fi_params,X)
    return s*jnp.exp(l)

P_Fi=jax.jit(lambda _,x: fi(x)**2)

state=TrainState.create(apply_fn=slog_Si,params=params0,tx=optax.sgd(.1,.2))
vals=[]
nburn=1000
proposal=mcmc.mcmc.gaussianproposal(.5)

s=250
Y=Y[:s]
ip,ic,(n1,n2)=[config.problem[k] for k in ['ion_pos','ion_charges','nelec']]
_,X=initialize_molecular_pos(mcmc.session.nextkey(),s,jnp.array(ip),jnp.array(ic),n1+n2)

match mode:
    case 'mcmc':
        E_value_and_grad=mcmc.energy.MCMC_value_and_grad(slog_Si,fi=fi)

        #X=rnd.normal(nextkey(),Y.shape)*std
        si_sampler=mcmc.mcmc.Metropolis(P=P_Si,proposal=proposal,walkers=X)
        fi_sampler=mcmc.mcmc.Metropolis(P=P_Fi,proposal=proposal,walkers=Y)

        try:
            for i in tqdm(range(10)):
                X=si_sampler.sample(params0,steps=100)
        
        except KeyboardInterrupt: pass

        try:
            for i in tqdm(range(1000)):
                print('sampling')
                X=si_sampler.sample(state.params,steps=100)
                Y=fi_sampler.sample(None)
                
                print('gradient')
                loss,grads=E_value_and_grad(state.params,X,Y)
                state=state.apply_gradients(grads=grads)
                vals.append(-loss)
                print(-loss)

        except KeyboardInterrupt: pass

        mcmc.session.save(vals,'outputs/mcmc')


    case 'correlated':
        E_value_and_grad=mcmc.energy.correlated_value_and_grad(slog_Si)

        fi_sampler=mcmc.mcmc.Metropolis(P=P_Fi,proposal=proposal,walkers=Y)
        #Y=fi_sampler.sample(None,steps=nburn)

        try:
            for i in tqdm(range(1000)):
                print('sampling')
                Y=fi_sampler.sample(None,10)
                fY=fi(Y)

                print('gradient')
                loss,grads=E_value_and_grad(state.params,Y,fY)
                state=state.apply_gradients(grads=grads)
                vals.append(-loss)
                print(-loss)
        except KeyboardInterrupt:
            pass

        mcmc.session.save(vals,'outputs/correlated')


    case 'no_mc':
        E_value_and_grad=mcmc.energy.NO_MC_value_and_grad(slog_Si,fi)

        for i in range(1000):
            #X=rnd.normal(nextkey(),Y.shape)*std
            loss,grads=E_value_and_grad(state.params,X)
            state=state.apply_gradients(grads=grads)
            vals.append(-loss)
            print(-loss)

        mcmc.session.save(vals,'outputs/no_mc')
