import os
import subprocess

def listpaths(path):
    return [os.path.join(path,f) for f in os.listdir(path)]

def pickfrom(alloptions,allinfos):
    filter=''
    while True:
        try:
            options,infos=zip(*[(o,i) for o,i in zip(alloptions,allinfos) if filter in o])
        except:
            options,infos=alloptions,allinfos
            print('no matches')
        optionstext='\n'.join(['{}: {} ({})'.format(i+1,p,info) for i,(p,info) in enumerate(zip(options,infos))])
        inp=input('\nInput number 1-{} to select OR input f0101- to filter by substring 0101- (i.e. Jan 1):\n\n{}\n'.format(len(options),optionstext))
        if inp[:1]=='f':
            filter=inp[1:]
        else:
            try:
                optionid=int(inp)-1
                return options[optionid]
            except:
                pass

def isrunpath(p):
    try:
        return all([f in os.listdir(p) for f in ['energy.txt','checkpoints','config.json']])
    except:
        return False

#
# input 20220101-120000 -> output 20220101-120000
# input logs directory -> browse logs directory
#
def pickrun(logspath):
    if isrunpath(logspath):
        return logspath
    paths=[d for d in reversed(sorted(listpaths(logspath))) if isrunpath(d)]
    assert(len(paths)>0)
    def getinfo(path):
        try:
            return '{} checkpoints'.format(len(os.listdir(os.path.join(path,'checkpoints'))))
        except:
            return ''
    infos=[getinfo(p) for p in paths]
    return pickfrom(paths,infos)

def showfile(path):
    try: subprocess.Popen(['open',path])
    except: pass
    try: subprocess.Popen(['xdg-open',path])
    except: pass
    try: os.startfile(path)
    except: pass
