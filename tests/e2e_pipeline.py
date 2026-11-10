"""
End-to-end check of the REAL main.py alert pipeline (python tests/e2e_pipeline.py).
Only FastAPI and the database are stubbed; the model, channels, timers, ntfy and SMS code are the real project code.
A fault reading goes in; it must come out as one ntfy request, one SMS and one saved alert.
"""
import time as _t
import sys, os, types, time, json, threading, http.server, io, contextlib
from unittest.mock import MagicMock
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT); sys.path.insert(0,os.path.join(ROOT,'tests')); sys.path.insert(0,os.path.join(ROOT,'tools'))

# ---- stub the web framework + database (everything else is the real project code) ----
class _R:
    def __init__(s,*a,**k): pass
    def _d(s,*a,**k): return lambda f: f
    get=post=put=delete=patch=middleware=_d
    def include_router(s,*a,**k): pass
    def mount(s,*a,**k): pass
    def add_middleware(s,*a,**k): pass
    def on_event(s,*a,**k): return lambda f: f
class HTTPExc(Exception):
    def __init__(s,status_code=500,detail=""): s.status_code=status_code; s.detail=detail
fa=types.ModuleType("fastapi")
for n,v in dict(FastAPI=_R,APIRouter=_R,HTTPException=HTTPExc,Request=object,Response=object,BackgroundTasks=object,Header=lambda *a,**k:None,Depends=lambda *a,**k:None,Query=lambda *a,**k:None).items(): setattr(fa,n,v)
sys.modules["fastapi"]=fa
for m in ("fastapi.responses","fastapi.staticfiles","fastapi.middleware","fastapi.middleware.cors"):
    sys.modules[m]=MagicMock()
pyd=types.ModuleType("pydantic"); 
class BM: 
    def __init__(s,**k): s.__dict__.update(k)
pyd.BaseModel=BM; pyd.Field=lambda *a,**k:None; sys.modules["pydantic"]=pyd

class Cur:
    def __init__(s): s.rowcount=0
    def execute(s,sql,params=None): s.sql=" ".join(sql.split())
    def fetchone(s): return {"id":1,"c":0}
    def fetchall(s): return []
class Conn:
    def cursor(s): return Cur()
    def commit(s): pass
    def rollback(s): pass
    def close(s): pass
DBMODE={"delay":0.0,"fail":False,"fail_save":False}
def _conn():
    if DBMODE["delay"]: _t.sleep(DBMODE["delay"])
    if DBMODE["fail"]: raise RuntimeError("database unavailable")
    return Conn()
db=types.ModuleType("database"); db.get_db_connection=_conn; sys.modules["database"]=db
acs=types.ModuleType("ac_setup"); acs.get_ac_setup=lambda: {"ac_status":"Cool","ac_thermostat":"22","data_gathering_mode":"telemetry","data_gathering_unit":"AC1"}; acs.update_ac_setup=lambda *a,**k:None; sys.modules["ac_setup"]=acs
wr=types.ModuleType("web_routes"); wr.router=_R(); wr._read_template=lambda *a,**k:""; wr.app_version=lambda *a,**k:"x"; sys.modules["web_routes"]=wr
for m in ("services.contacts_api","services.traccar_setup_api","services.ai_setup_api","services.auth","services.ai_predictor","services.traccar_sms"):
    mod=MagicMock(); mod.router=_R(); sys.modules[m]=mod
sys.modules["services.ai_predictor"].ask=lambda p: (_ for _ in ()).throw(RuntimeError("no LLM"))
CAP={"alerts":[],"sms":[]}
al=types.ModuleType("services.alerts_api"); al.router=_R(); al.AlertIn=lambda summary,diagnoses=None: types.SimpleNamespace(summary=summary,diagnoses=diagnoses)
def _create(a):
    if DBMODE["fail_save"] or DBMODE["fail"]: raise RuntimeError("cannot save the alert")
    CAP["alerts"].append(a); return types.SimpleNamespace(id=len(CAP["alerts"]))
al.create_alert=_create; al.update_alert_diagnoses=lambda i,d:None; sys.modules["services.alerts_api"]=al
sys.modules["services.traccar_sms"].send_sms=lambda t: CAP["sms"].append(t)

# ---- fake ntfy server ----
SEEN=[]
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(s):
        b=s.rfile.read(int(s.headers.get("Content-Length",0))); SEEN.append((s.path,dict(s.headers),b.decode(),_t.time()))
        s.send_response(200); s.send_header("Content-Length","2"); s.end_headers(); s.wfile.write(b"{}")
    def log_message(*a): pass
srv=http.server.HTTPServer(("127.0.0.1",0),H); threading.Thread(target=srv.serve_forever,daemon=True).start()
os.environ["NTFY_TOPIC"]="airbnnrr_______testttt"; os.environ["NTFY_SERVER"]=f"http://127.0.0.1:{srv.server_address[1]}"
os.environ.setdefault("SUPABASE_URL","x")

buf=io.StringIO()
with contextlib.redirect_stdout(buf):
    import main
    import fault_data as fd
    from services import ml_per_ac
print("main.py imported OK")
bundle=ml_per_ac.train_unit(fd.normal_rows(4000))
main.detector._bundles={"AC1":bundle}; main.detector._loaded_at=time.time()+10**9

fake_now=[1_000_000.0]
class T:  # controllable clock for the alert timers
    @staticmethod
    def time(): return fake_now[0]
    sleep=staticmethod(_t.sleep)
    strftime=staticmethod(_t.strftime)
main.time=T
import services.anomaly_reciever as ar; ar.time=T

def feed(r, wait=True):
    p=dict(r); p["ac_unit"]="AC1"; p["_received_at"]=T.time()
    with contextlib.redirect_stdout(buf):
        main.process_telemetry_background(p)

def settle(n_ntfy, n_sms, timeout=5.0):
    end=_t.time()+timeout
    while _t.time()<end and not (len(SEEN)>=n_ntfy and len(CAP["sms"])>=n_sms): _t.sleep(0.02)

def reset():
    main.scheduler.reset(); SEEN.clear(); CAP["alerts"].clear(); CAP["sms"].clear()

fault=fd.scenarios()["sim_fault (over every limit)"]
normal=fd.normal_rows(50,seed=3,start=3e6).to_dict("records")

# 0. warm-up with a normal reading: no alert, caches filled
feed(normal[0]); _t.sleep(0.2)
assert not SEEN and not CAP["sms"], "a normal reading must not alert"

# 1. healthy database. First fault reading: the INSTANT SAFETY alert leaves at once. The second consecutive severe reading
#    confirms the INSTANT ML alert (one reading alone could be a sensor glitch).
reset(); t0=_t.time(); feed(fault[0]); settle(1,1)
assert [h["Title"] for _,h,_,_ in SEEN]==["STOP (Safety): AC1"], [h["Title"] for _,h,_,_ in SEEN]
feed(fault[1]); settle(2,2)
titles=sorted(h["Title"] for _,h,_,_ in SEEN)
assert titles==["STOP (ML): AC1","STOP (Safety): AC1"], titles
assert len(CAP["sms"])==2 and len(CAP["alerts"])==2
assert all("[ML]" in m or "[SAFETY]" in m for m in CAP["sms"])
print(f"1. fault: Safety alert on the 1st reading, ML alert on the 2nd -> {len(SEEN)} ntfy + {len(CAP['sms'])} SMS + {len(CAP['alerts'])} saved in {_t.time()-t0:.2f}s: {titles}")

# 2. SLOW database (2 s per connection): the alert must still leave at once; storing the reading waits its turn AFTER it
reset(); DBMODE["delay"]=2.0
t0=_t.time(); th=threading.Thread(target=feed,args=(fault[2],)); th.start()
settle(1,1,timeout=1.5); first=min(a[3] for a in SEEN)-t0 if SEEN else None
th.join(); total=_t.time()-t0; DBMODE["delay"]=0.0
assert first is not None and first<0.5, f"alert took {first} s with a slow database"
assert total>=1.9, "storing the reading (slow) must happen after the alert"
assert len(SEEN)==1 and len(CAP["sms"])==1
print(f"2. slow database: first ntfy after {first:.2f}s while storing the reading took {total:.2f}s")

# 3. FAILING database: the alerts still go out (before, a failed insert meant no alert at all)
reset(); DBMODE["fail"]=True
feed(fault[3]); feed(fault[4]); settle(2,2); DBMODE["fail"]=False
assert len(SEEN)==2, "ntfy must not depend on the database"
assert len(CAP["sms"])==2, "the SMS must not depend on the database (sent without a Report ID)"
assert all("Report ID" not in m for m in CAP["sms"])
print("3. database down: both ntfy alerts and both SMS still delivered")

# 4. a NEW instant problem (the old one cleared) is not held back for the full 2-minute pause
reset(); fake_now[0]+=1000
feed(fault[5]); settle(1,1)
for i in range(8):                                   # 80 s of normal readings: the problem is over
    fake_now[0]+=10; feed(normal[i+1])
fake_now[0]+=10; before=len(SEEN); feed(fault[6]); settle(before+1,len(CAP["sms"])+1)
assert len(SEEN)>=before+1, "a new fault 90 s after the last one must alert at once, not wait out the repeat pause"
print("4. second fault 90 s after the first (in between: normal) alerted immediately")

# 5. a single severe glitch reading between normal ones must NOT produce an instant ML alert
reset(); fake_now[0]+=1000
for i in range(12): fake_now[0]+=10; feed(normal[i])
fake_now[0]+=10; feed(fault[7]); 
for i in range(12,20): fake_now[0]+=10; feed(normal[i])
_t.sleep(0.4)
assert not any("(ML)" in h["Title"] for _,h,_,_ in SEEN), "one glitch reading must not be an instant ML alert"
print("5. a single glitch reading did not trigger an instant ML alert")

st=main.ntfy_notifier.status()
assert st["last_ok"] is True and st["failed"]==0
print("\nE2E OK: real main.py -> channels -> ntfy + SMS + saved alert (healthy, slow and failing database)")
