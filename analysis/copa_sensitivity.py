import sqlite3, sys, time, json
sys.path.insert(0, '.')
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from src import adapters
from src.runner import build_prompt, load_templates, load_samples

tpl = load_templates(); items = load_samples("copa")[:200]
mt = tpl["max_tokens"]["copa"]
db = sqlite3.connect("data/responses.db")
db.execute("""CREATE TABLE IF NOT EXISTS copa_t07 (model TEXT,level INT,item_id TEXT,sample_idx INT,response TEXT,error TEXT,
 PRIMARY KEY(model,level,item_id,sample_idx))""")
done={(l,i,s) for l,i,s in db.execute("SELECT level,item_id,sample_idx FROM copa_t07 WHERE model='exaone3.5' AND response IS NOT NULL")}
jobs=[(lv,it,s) for lv in (1,3,5) for it in items for s in range(3) if (lv,str(it["item_id"]),s) not in done]
print(f"COPA 민감도 대상 {len(jobs)}건", flush=True)
def run(j):
    lv,it,s=j; p=build_prompt(tpl,"copa",lv,1,it)
    try: r=adapters.generate("exaone3.5",p,max_tokens=mt,temperature=0.7,seed=42+s); e=None
    except Exception as ex: r=None; e=str(ex)
    return lv,it,s,r,e
t0=time.time(); n=0
with ThreadPoolExecutor(4) as ex:
    for f in as_completed([ex.submit(run,j) for j in jobs]):
        lv,it,s,r,e=f.result()
        db.execute("INSERT OR REPLACE INTO copa_t07 VALUES(?,?,?,?,?,?)",("exaone3.5",lv,str(it["item_id"]),s,r,e))
        n+=1
        if n%100==0: db.commit(); print(f"{n}/{len(jobs)}",flush=True)
db.commit(); print(f"완료 {n}건 {time.time()-t0:.0f}s")
