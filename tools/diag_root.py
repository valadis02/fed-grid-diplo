# Διάγνωση root FLTrust: μέση cosine κάθε σπιτιού με τα υπόλοιπα του fog (γύροι 1-3, χωρίς επίθεση). Επιλέγει medoid εκτός επιτιθέμενων.
# Χρήση: python tools/diag_root.py
import os,sys
ROOT_=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])),'..'))
sys.argv=['x','--strategy','fedavg','--data','lcl50_similar']; sys.path.insert(0,ROOT_); __file__=os.path.join(ROOT_,'tools','offline_sim.py')
exec(open(os.path.join(ROOT_,'tools','offline_sim.py')).read().split('def main():')[0])
import numpy as np, json
import fog.model.fog_attacks as fa
from shared.attackers import attackers_for
H=list(range(1,51)); data={i:load_house(i) for i in H}; globals()['data']=data; globals()['ATT']=[]
model=create_model(os.getenv("MODEL_ARCHITECTURE","simple_lstm_two_gates")); w=[x.copy() for x in model.get_weights()]
ever=set(attackers_for(40))
out={}
for r in range(1,4):
    G={}
    for i in H:
        A=fc.build_arrays(data[i][0],data[i][1]); wi=local_train(model,w,A,1000*r+i); G[i]=fa._flatten(wi)-fa._flatten(w)
    res={}
    for k,mem in ((1,H[:25]),(2,H[25:])):
        M=np.stack([G[i] for i in mem]); U=M/np.linalg.norm(M,axis=1,keepdims=True); C=U@U.T
        meancos={h:float((C[j].sum()-1)/24) for j,h in enumerate(mem)}
        zeros={h:int((C[j]<=0).sum()) for j,h in enumerate(mem)}
        res[k]=dict(meancos=meancos,zeros=zeros)
        best=sorted(mem,key=lambda h:-meancos[h])
        print(f"r{r} fog{k} | root1/26 meancos={meancos[mem[0]]:.3f} honest-with-cos<=0 if root={zeros[mem[0]]}/25 | medoid order(not attacker):",
              [(h,round(meancos[h],3),zeros[h]) for h in best if h not in ever][:4], "| worst:",[(h,round(meancos[h],3)) for h in best[-3:]],flush=True)
    out[r]=res
    # FedAvg global for next round (as cloud: mean(prev, fog1, fog2))
    f1=np.mean([G[i] for i in H[:25]],0); f2=np.mean([G[i] for i in H[25:]],0); base=fa._flatten(w)
    w=fa._unflatten(base+(f1+f2)/3,w)
json.dump(out,open('diag_root.json','w'))

