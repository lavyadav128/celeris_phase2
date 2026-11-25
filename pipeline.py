
from pathlib import Path
import json, warnings, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from lightgbm import LGBMRegressor
warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"data"; OUT=ROOT/"outputs"; OUT.mkdir(exist_ok=True)

def norm_month(s):
    s=s.astype(str).str.strip()
    a=pd.to_datetime(s,format="%Y-%m",errors="coerce")
    b=pd.to_datetime(s,format="%m-%Y",errors="coerce")
    c=pd.to_datetime(s,format="mixed",errors="coerce")
    return a.fillna(b).fillna(c).dt.to_period("M").dt.to_timestamp()

def read_inputs():
    return {p.stem:pd.read_csv(p,low_memory=False) for p in DATA.glob("*.csv")}

def add_event_features(frame, calendar):
    frame=frame.copy()
    dates=frame.Date.to_numpy(); region=frame.Region.astype(str).to_numpy(); family=frame.Product_Family.astype(str).to_numpy()
    channel=frame.Sales_Channel.astype(str).to_numpy(); customer=frame.Customer_Name.astype(str).to_numpy()
    count=np.zeros(len(frame),dtype=np.int16); disruption=np.zeros(len(frame),dtype=np.int8)
    price=np.zeros(len(frame),dtype=np.int8); supply=np.zeros(len(frame),dtype=np.int8)
    for _,ev in calendar.iterrows():
        start=pd.to_datetime(str(ev.get("Start_Date",""))[:7]+"-01",errors="coerce")
        endraw=ev.get("End_Date",None)
        end=pd.to_datetime((str(endraw)[:7] if pd.notna(endraw) and str(endraw).strip() else "2099-12")+"-01",errors="coerce")
        if pd.isna(start) or pd.isna(end): continue
        mask=(dates>=np.datetime64(start))&(dates<=np.datetime64(end))
        scope=str(ev.get("Scope_Type","Global")).strip().lower(); val=str(ev.get("Scope_Value","")).strip()
        if scope=="region": mask &= region==val
        elif scope=="product_family": mask &= family==val
        elif scope=="channel": mask &= channel==val
        elif scope=="customer": mask &= customer==val
        elif scope!="global": continue
        count += mask.astype(np.int16)
        typ=str(ev.get("Event_Type","")).lower()
        if "disrupt" in typ or "logistic" in typ: disruption=np.maximum(disruption,mask.astype(np.int8))
        if "price" in typ: price=np.maximum(price,mask.astype(np.int8))
        if "supply" in typ: supply=np.maximum(supply,mask.astype(np.int8))
    frame["Event_Count"]=count; frame["Event_Disruption"]=disruption
    frame["Event_Price_Change"]=price; frame["Event_Supply_Constraint"]=supply
    return frame

def clean(d):
    train=d["train"].copy(); test=d["test"].copy()
    issues=[]
    train["Date_raw"]=train["Date"]
    train["Date"]=norm_month(train["Date"])
    test["Date"]=norm_month(test["Date"])
    bad_date=int(train.Date.isna().sum())
    issues.append(["invalid_date","train.csv",bad_date,"Rows with unparseable Date removed"])
    train=train.dropna(subset=["Date"])
    # Date is authoritative
    train["Year"]=train.Date.dt.year; train["Month"]=train.Date.dt.month
    test["Year"]=test.Date.dt.year; test["Month"]=test.Date.dt.month
    # normalize text labels
    textcols=["Product_Family","Business_Unit","SKU_Description","Customer_Name","Sales_Channel","Region","SKU_ID"]
    for c in textcols:
        train[c]=train[c].astype("string").str.strip()
        test[c]=test[c].astype("string").str.strip()
        train[c]=train[c].replace({"":pd.NA,"nan":pd.NA,"None":pd.NA})
        test[c]=test[c].replace({"":pd.NA,"nan":pd.NA,"None":pd.NA})
    # fill master attributes from sku master
    master=d["sku_master"].copy()
    for c in ["SKU_ID","Product_Family","Business_Unit","SKU_Description"]:
        master[c]=master[c].astype("string").str.strip()
    for c in ["Product_Family","Business_Unit","SKU_Description"]:
        mp=master.drop_duplicates("SKU_ID").set_index("SKU_ID")[c]
        train[c]=train[c].fillna(train.SKU_ID.map(mp))
        test[c]=test[c].fillna(test.SKU_ID.map(mp))
    # stable de-duplication
    train["Load_Timestamp_parsed"]=pd.to_datetime(train.Load_Timestamp,errors="coerce")
    before=len(train)
    train=train.sort_values("Load_Timestamp_parsed").drop_duplicates("Record_ID",keep="last")
    issues.append(["duplicate_record_id","train.csv",before-len(train),"Kept latest Load_Timestamp per Record_ID"])
    before=len(train)
    train=train.sort_values("Load_Timestamp_parsed").drop_duplicates(["Date","SKU_ID","Region"],keep="last")
    issues.append(["duplicate_date_sku_region","train.csv",before-len(train),"Kept latest loaded row per monthly key"])
    # numeric cleanup: comma-separated thousands accepted
    for c in ["Units_Sold","Unit_Price_USD","Revenue_USD"]:
        train[c]=pd.to_numeric(train[c].astype(str).str.replace(",","",regex=False),errors="coerce")
    # repair obvious negative / missing targets as unknown, not zero
    bad_units=int((train.Units_Sold<0).sum())
    train.loc[train.Units_Sold<0,"Units_Sold"]=np.nan
    issues.append(["negative_units","train.csv",bad_units,"Set invalid negative target to missing"])
    # Stockout file gives requested units as unconstrained demand
    so=d["inventory_stockouts"].copy(); so["Date"]=norm_month(so.Date)
    so=so.dropna(subset=["Date"]).drop_duplicates(["Date","SKU_ID","Region"],keep="last")
    so["Demand_Observed"]=pd.to_numeric(so.Requested_Units,errors="coerce")
    train=train.merge(so[["Date","SKU_ID","Region","Demand_Observed","Shipped_Units","Fill_Rate_Pct","Reason"]],
                      on=["Date","SKU_ID","Region"],how="left")
    train["Demand_Unconstrained"]=train.Demand_Observed.where(train.Demand_Observed.notna(),train.Units_Sold)
    train["Stockout_Adjusted"]=train.Demand_Observed.notna()
    # price validation; use SKU median valid price, fallback family median
    valid_price=train.loc[train.Unit_Price_USD.between(0.01,100000),"Unit_Price_USD"]
    sku_price=train.loc[train.Unit_Price_USD.between(0.01,100000)].groupby("SKU_ID").Unit_Price_USD.median()
    fam_price=train.loc[train.Unit_Price_USD.between(0.01,100000)].groupby("Product_Family").Unit_Price_USD.median()
    bad_price=int((~train.Unit_Price_USD.between(0.01,100000)).sum())
    train["Unit_Price_Clean"]=train.Unit_Price_USD.where(train.Unit_Price_USD.between(0.01,100000))
    train["Unit_Price_Clean"]=train.Unit_Price_Clean.fillna(train.SKU_ID.map(sku_price)).fillna(train.Product_Family.map(fam_price))
    issues.append(["invalid_price","train.csv",bad_price,"Replaced zero/extreme prices using SKU median, then family median"])
    train["Revenue_Recalc"]=train.Demand_Unconstrained*train.Unit_Price_Clean
    mismatch=int(((train.Revenue_USD-train.Units_Sold*train.Unit_Price_USD).abs()>1).fillna(False).sum())
    issues.append(["revenue_reconciliation","train.csv",mismatch,"Recalculated clean revenue from unconstrained demand and cleaned price"])
    # missing key fields are not silently converted to zero
    train=train.dropna(subset=["Date","SKU_ID","Region"])
    test=test.dropna(subset=["Date","SKU_ID","Region"])
    # lifecycle enforcement on historical observed targets
    for c in ["Launch_Date","EOL_Date"]:
        master[c+"_dt"]=norm_month(master[c].fillna(""))
    life=master.set_index("SKU_ID")
    launch=train.SKU_ID.map(life.Launch_Date_dt)
    eol=train.SKU_ID.map(life.EOL_Date_dt)
    impossible=((launch.notna())&(train.Date<launch))|((eol.notna())&(train.Date>eol))
    issues.append(["lifecycle_conflict","train.csv",int(impossible.sum()),"Flagged out-of-lifecycle records; excluded from model targets"])
    train["Lifecycle_Conflict"]=impossible
    train.loc[impossible,"Demand_Unconstrained"]=np.nan
    # Event-calendar feature engineering: only scoped, date-overlapping events are activated.
    events=d["events_calendar"].copy()
    events["_start"]=pd.to_datetime(events["Start_Date"].astype(str).str[:7]+"-01",errors="coerce")
    end_raw=events["End_Date"].fillna("2099-12").astype(str)
    events["_end"]=pd.to_datetime(end_raw.str[:7]+"-01",errors="coerce")
    train=add_event_features(train,events)
    # macro date and numeric cleaning
    macro=d["macro_indicators"].copy(); macro["Date"]=norm_month(macro.Date)
    macro=macro.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date",keep="last")
    for c in macro.columns:
        if c not in ["Date"]: macro[c]=pd.to_numeric(macro[c],errors="coerce")
    macro=macro.sort_values("Date").ffill().bfill()
    # anomaly flags using robust SKU-level threshold
    grp=train.groupby("SKU_ID").Demand_Unconstrained
    med=grp.transform("median"); mad=grp.transform(lambda x: np.nanmedian(np.abs(x-np.nanmedian(x))))
    robust_z=(train.Demand_Unconstrained-med)/(1.4826*mad.replace(0,np.nan))
    train["Extreme_Demand_Flag"]=robust_z.abs()>8
    issues.append(["extreme_demand","train.csv",int(train.Extreme_Demand_Flag.sum()),"Flagged robust SKU-level outliers; retained pending event review"])
    train.to_csv(OUT/"cleaned_train.csv",index=False)
    test.to_csv(OUT/"cleaned_test.csv",index=False)
    macro.to_csv(OUT/"cleaned_macro.csv",index=False)
    issue_df.to_csv(OUT/"data_quality_report.csv",index=False)
    return train,test,macro,master,d["events_calendar"],issue_df

CAT=["SKU_ID","Region","Product_Family","Customer_Name","Sales_Channel","Business_Unit","Powertrain_Application","Lifecycle_Status","Intermittency_Class"]
def make_panel(train,test,master,macro,end_date,include_future=False,events=None):
    keys=test[["SKU_ID","Region","Product_Family","Business_Unit","Customer_Name","Sales_Channel"]].drop_duplicates()
    # history for key combinations in test plus all available combinations
    hist=train[["Date","SKU_ID","Region","Demand_Unconstrained","Unit_Price_Clean","Event_Count","Event_Disruption","Event_Price_Change","Event_Supply_Constraint"]+[
        c for c in ["Product_Family","Business_Unit","Customer_Name","Sales_Channel"]]].copy()
    hist=hist.sort_values("Date")
    # restrict to series relevant for requested forecast keys
    key_idx=pd.MultiIndex.from_frame(keys[["SKU_ID","Region"]])
    hist=hist.set_index(["SKU_ID","Region"])
    hist=hist.loc[hist.index.isin(key_idx)].reset_index()
    starts=hist.groupby(["SKU_ID","Region"]).Date.min().rename("start").reset_index()
    base=keys.merge(starts,on=["SKU_ID","Region"],how="left")
    base["start"]=base.start.fillna(pd.Timestamp("2019-01-01"))
    grids=[]
    for r in base.itertuples(index=False):
        ds=pd.date_range(r.start,end_date,freq="MS")
        grids.append(pd.DataFrame({"Date":ds,"SKU_ID":r.SKU_ID,"Region":r.Region,
            "Product_Family":r.Product_Family,"Business_Unit":r.Business_Unit,
            "Customer_Name":r.Customer_Name,"Sales_Channel":r.Sales_Channel}))
    panel=pd.concat(grids,ignore_index=True)
    panel=panel.merge(hist[["Date","SKU_ID","Region","Demand_Unconstrained","Unit_Price_Clean","Event_Count","Event_Disruption","Event_Price_Change","Event_Supply_Constraint"]],on=["Date","SKU_ID","Region"],how="left")
    # SKU master metadata
    m=master.drop_duplicates("SKU_ID").copy()
    panel=panel.merge(m[["SKU_ID","Powertrain_Application","Lifecycle_Status","Launch_Date","EOL_Date","Replaces_SKU"]],on="SKU_ID",how="left")
    for c in ["Launch_Date","EOL_Date"]: panel[c+"_dt"]=norm_month(panel[c].fillna(""))
    panel=panel.sort_values(["SKU_ID","Region","Date"]).reset_index(drop=True)
    for c in ["Event_Count","Event_Disruption","Event_Price_Change","Event_Supply_Constraint"]: panel[c]=panel[c].fillna(0)
    panel["observed"]=panel.Demand_Unconstrained.notna().astype(int)
    panel["month"]=panel.Date.dt.month; panel["year"]=panel.Date.dt.year
    panel["month_sin"]=np.sin(2*np.pi*panel.month/12); panel["month_cos"]=np.cos(2*np.pi*panel.month/12)
    panel["time_idx"]=(panel.Date.dt.year-2019)*12+panel.Date.dt.month
    panel["age_months"]=((panel.Date.dt.year-panel.Launch_Date_dt.dt.year)*12+(panel.Date.dt.month-panel.Launch_Date_dt.dt.month)).fillna(0).clip(lower=0)
    panel["is_after_eol"]=((panel.EOL_Date_dt.notna())&(panel.Date>panel.EOL_Date_dt)).astype(int)
    panel["is_before_launch"]=((panel.Launch_Date_dt.notna())&(panel.Date<panel.Launch_Date_dt)).astype(int)
    # macro indicators and lags
    mac=macro.copy()
    mac_cols=[c for c in mac.columns if c not in ["Date","Year","Month"]]
    for lag in [0,1,2,3]:
        tmp=mac[["Date"]+mac_cols].copy()
        tmp["Date"]=tmp.Date+pd.DateOffset(months=lag)
        tmp=tmp.rename(columns={c:f"{c}_lag{lag}" for c in mac_cols})
        panel=panel.merge(tmp,on="Date",how="left")
    print("  merged macros",panel.shape,flush=True)
    # historical lags and rolling stats computed per series
    g=panel.groupby(["SKU_ID","Region"],sort=False)["Demand_Unconstrained"]
    for lag in [1,2,3,6,12]:
        panel[f"lag_{lag}"]=g.shift(lag)
    shifted=g.shift(1)
    grouped_shift=shifted.groupby([panel.SKU_ID,panel.Region],sort=False)
    for w in [3,6,12]:
        panel[f"roll_mean_{w}"]=grouped_shift.rolling(w,min_periods=1).mean().reset_index(level=[0,1],drop=True).sort_index()
        panel[f"roll_std_{w}"]=grouped_shift.rolling(w,min_periods=2).std().reset_index(level=[0,1],drop=True).sort_index()
    panel["nonzero_rate_12"]=shifted.gt(0).groupby([panel.SKU_ID,panel.Region],sort=False).rolling(12,min_periods=1).mean().reset_index(level=[0,1],drop=True).sort_index()
    panel["months_since_nonzero"]=0
    predecessor=train[["Date","SKU_ID","Region","Demand_Unconstrained"]].rename(columns={"SKU_ID":"Replaces_SKU","Demand_Unconstrained":"Replacement_Demand"})
    panel=panel.merge(predecessor[predecessor.Replaces_SKU.notna()],on=["Date","Region","Replaces_SKU"],how="left")
    repstats=train.sort_values("Date").groupby(["SKU_ID","Region"]).tail(12).groupby(["SKU_ID","Region"]).Demand_Unconstrained.mean().rename("Replacement_Recent_Mean").reset_index().rename(columns={"SKU_ID":"Replaces_SKU"})
    panel=panel.merge(repstats,on=["Replaces_SKU","Region"],how="left")
    stats=panel.groupby(["SKU_ID","Region"]).Demand_Unconstrained.agg(total_months="count",nonzero=lambda x:int((x>0).sum()),mean="mean",std="std").reset_index()
    stats["ADI"]=stats.total_months/stats.nonzero.replace(0,np.nan)
    stats["CV2"]=(stats["std"].fillna(0)/stats["mean"].replace(0,np.nan))**2
    stats["Intermittency_Class"]=np.select([stats.ADI<=1.32,stats.CV2<=0.49], ["Smooth","Intermittent"],default="Lumpy")
    stats.loc[stats.nonzero.eq(0),"Intermittency_Class"]="No observed demand"
    panel=panel.merge(stats[["SKU_ID","Region","ADI","CV2","Intermittency_Class"]],on=["SKU_ID","Region"],how="left")
    return panel,mac_cols

def train_model(panel, cutoff=None, objective="tweedie", alpha=0.5):
    data=panel[panel.Demand_Unconstrained.notna()].copy()
    if cutoff is not None: data=data[data.Date<=cutoff]
    feature_cols=[c for c in panel.columns if c.startswith(("lag_","roll_","nonzero_rate","Event_","Replacement_Demand","Replacement_Recent_Mean","ADI","CV2","month_sin","month_cos","time_idx","age_months","is_after_eol","is_before_launch"))]
    feature_cols += [c for c in panel.columns if "_lag" in c and c not in feature_cols]
    cats=[c for c in CAT if c in panel.columns]
    vocab={c:pd.Index(panel[c].astype("string").fillna("__MISSING__").unique()) for c in cats}
    X=data[feature_cols+cats].copy()
    for c in cats:
        X[c]=pd.Categorical(data[c].astype("string").fillna("__MISSING__"),categories=vocab[c]).codes
    X=X.replace([np.inf,-np.inf],np.nan).fillna(-1)
    y=data.Demand_Unconstrained.clip(lower=0)
    model=LGBMRegressor(n_estimators=45,learning_rate=.035,num_leaves=31,max_depth=-1,
        subsample=.85,colsample_bytree=.85,reg_lambda=2,objective=objective,
        alpha=alpha,tweedie_variance_power=1.3,random_state=42,n_jobs=4,verbosity=-1)
    model.fit(X,y)
    return model,feature_cols,cats



def forecast_recursive(train,test,master,macro,model,feature_cols,cats,events=None):
    print("Building forecast panel...",flush=True)
    panel,mac_cols=make_panel(train,test,master,macro,pd.Timestamp("2028-05-01"))
    print("Forecast panel ready",flush=True)
    vocab={c:pd.Index(panel[c].astype("string").fillna("__MISSING__").unique()) for c in cats}
    panel=panel.sort_values(["SKU_ID","Region","Date"]).reset_index(drop=True)
    hist=panel[panel.Date<pd.Timestamp("2026-11-01")]
    future=panel[panel.Date>=pd.Timestamp("2026-11-01")].copy()
    if events is not None: future=add_event_features(future,events)
    # Direct 19-month strategy: use the latest observed history and stable rolling profile,
    # while month/year/macro/lifecycle features vary for each horizon month.

    state_rows=[]
    for (sku,reg),g in hist.groupby(["SKU_ID","Region"],sort=False):
        a=g.Demand_Unconstrained.to_numpy(dtype=float)
        row={"SKU_ID":sku,"Region":reg}
        for lag in [1,2,3,6,12]:
            row[f"lag_{lag}"]=a[-lag] if len(a)>=lag else np.nan
        for w in [3,6,12]:
            z=a[-w:]
            row[f"roll_mean_{w}"]=np.nanmean(z) if np.isfinite(z).any() else np.nan
            row[f"roll_std_{w}"]=np.nanstd(z,ddof=1) if np.isfinite(z).sum()>1 else np.nan
        z=a[-12:]
        row["nonzero_rate_12"]=np.mean(z>0) if len(z) else 0
        state_rows.append(row)
    state=pd.DataFrame(state_rows)
    future=future.drop(columns=[c for c in future.columns if c.startswith("lag_") or c.startswith("roll_") or c=="nonzero_rate_12"],errors="ignore")
    future=future.merge(state,on=["SKU_ID","Region"],how="left")
    X=future[feature_cols+cats].copy()
    for c in cats:
        X[c]=pd.Categorical(X[c].astype("string").fillna("__MISSING__"),categories=vocab[c]).codes
    X=X.replace([np.inf,-np.inf],np.nan).fillna(-1)
    print("Predicting",X.shape,flush=True)
    if isinstance(model,dict):
        q10=np.maximum(0,model[0.1].predict(X)); q50=np.maximum(0,model[0.5].predict(X)); q90=np.maximum(0,model[0.9].predict(X))
        # Enforce non-crossing quantiles.
        stack=np.sort(np.vstack([q10,q50,q90]),axis=0); q10,q50,q90=stack[0],stack[1],stack[2]
    else:
        q50=np.maximum(0,model.predict(X)); q10=np.maximum(0,q50*.65); q90=q50*1.35
    print("Predicted",flush=True)
    inactive=(future.is_after_eol.to_numpy()==1)|(future.is_before_launch.to_numpy()==1)
    q10[inactive]=0; q50[inactive]=0; q90[inactive]=0
    future["Forecast"]=q50
    future["P10"]=q10; future["P50"]=q50; future["P90"]=q90
    return future

def main():
    print("Loading inputs...",flush=True)
    d=read_inputs()
    print("Cleaning...",flush=True)
    train,test,macro,master,events,issues=clean(d)
    print("Cleaned",len(train),flush=True)
    # Build model panel through Oct-2026
    print("Building panel...",flush=True)
    panel,mac_cols=make_panel(train,test,master,macro,pd.Timestamp("2026-10-01"),events=events)
    print("Panel",panel.shape,flush=True)
    # Backtest: fit through Oct-2025, predict observed Nov-2025..Oct-2026 one-step (lags use actual prior months)
    print("Training validation model...",flush=True)
    model,feat,cats=train_model(panel,pd.Timestamp("2025-10-01"))
    print("Validation model done",flush=True)
    val=panel[(panel.Date>="2025-11-01")&(panel.Date<="2026-10-01")&panel.Demand_Unconstrained.notna()].copy()
    X=val[feat+cats].copy()
    for c in cats:
        vocab=pd.Index(panel[c].astype("string").fillna("__MISSING__").unique())
        X[c]=pd.Categorical(val[c].astype("string").fillna("__MISSING__"),categories=vocab).codes
    X=X.replace([np.inf,-np.inf],np.nan).fillna(-1)
    val["prediction"]=np.maximum(0,model.predict(X))
    val["baseline_seasonal_naive"]=val["lag_12"].fillna(val["roll_mean_12"]).fillna(0).clip(lower=0)
    val["baseline_recent_mean"]=val["roll_mean_3"].fillna(val["roll_mean_6"]).fillna(0).clip(lower=0)
    metric_rows=[]
    for label,col in [("Seasonal naive (lag 12)","baseline_seasonal_naive"),
                      ("Recent rolling mean (3 months)","baseline_recent_mean"),
                      ("LightGBM global Tweedie","prediction")]:
        err=np.abs(val.Demand_Unconstrained-val[col])
        metric_rows.append({"model":label,"backtest_window":"2025-11 to 2026-10",
            "WAPE":err.sum()/max(val.Demand_Unconstrained.sum(),1),
            "Bias":(val[col].sum()-val.Demand_Unconstrained.sum())/max(val.Demand_Unconstrained.sum(),1),
            "MAE":mean_absolute_error(val.Demand_Unconstrained,val[col]),"rows":len(val)})
    metrics=pd.DataFrame(metric_rows)
    model_row=metrics[metrics.model=="LightGBM global Tweedie"].iloc[0]
    wape=float(model_row.WAPE); bias=float(model_row.Bias)
    metrics.to_csv(OUT/"backtest_metrics.csv",index=False)
    val[["Date","SKU_ID","Region","Demand_Unconstrained","baseline_seasonal_naive","baseline_recent_mean","prediction"]].to_csv(OUT/"backtest_predictions.csv",index=False)
    origin=pd.Timestamp("2024-03-01"); vstart=pd.Timestamp("2024-04-01"); vend=pd.Timestamp("2025-10-01")
    model19,feat19,cats19=train_model(panel,origin)
    hist19=panel[panel.Date<=origin]; rows=[]
    for (sku,reg),g in hist19.groupby(["SKU_ID","Region"],sort=False):
        a=g.sort_values("Date").Demand_Unconstrained.to_numpy(dtype=float); r={"SKU_ID":sku,"Region":reg}
        for lag in [1,2,3,6,12]: r[f"lag_{lag}"]=a[-lag] if len(a)>=lag else np.nan
        for w in [3,6,12]:
            z=a[-w:]; r[f"roll_mean_{w}"]=np.nanmean(z) if np.isfinite(z).any() else np.nan
            r[f"roll_std_{w}"]=np.nanstd(z,ddof=1) if np.isfinite(z).sum()>1 else np.nan
        z=a[-12:]; r["nonzero_rate_12"]=np.mean(z>0) if len(z) else 0; rows.append(r)
    state19=pd.DataFrame(rows); v19=panel[(panel.Date>=vstart)&(panel.Date<=vend)&panel.Demand_Unconstrained.notna()].copy()
    v19=v19.drop(columns=[c for c in v19.columns if c.startswith("lag_") or c.startswith("roll_") or c=="nonzero_rate_12"],errors="ignore").merge(state19,on=["SKU_ID","Region"],how="left")
    X19=v19[feat19+cats19].copy()
    for c in cats19: X19[c]=pd.Categorical(v19[c].astype("string").fillna("__MISSING__"),categories=pd.Index(panel[c].astype("string").fillna("__MISSING__").unique())).codes
    X19=X19.replace([np.inf,-np.inf],np.nan).fillna(-1); v19["prediction"]=np.maximum(0,model19.predict(X19))
    e19=np.abs(v19.Demand_Unconstrained-v19.prediction); w19=float(e19.sum()/max(v19.Demand_Unconstrained.sum(),1)); b19=float((v19.prediction.sum()-v19.Demand_Unconstrained.sum())/max(v19.Demand_Unconstrained.sum(),1))
    pd.DataFrame([{"validation":"19-month fixed-origin holdout","origin":origin.strftime("%Y-%m"),"window":f"{vstart:%Y-%m} to {vend:%Y-%m}","WAPE":w19,"Bias":b19,"MAE":mean_absolute_error(v19.Demand_Unconstrained,v19.prediction),"rows":len(v19)}]).to_csv(OUT/"rolling_origin_19m_metrics.csv",index=False)
    # Fit final model on all observed history
    print("Training final model...",flush=True)
    final_model={}
    for alpha in [0.1,0.5,0.9]:
        print(f"Training quantile model {alpha}...",flush=True)
        final_model[alpha],feat,cats=train_model(panel,objective="quantile",alpha=alpha)
    print("Final quantile models done",flush=True)
    print("Forecasting...",flush=True)
    f=forecast_recursive(train,test,master,macro,final_model,feat,cats,events)
    print("Forecast done",flush=True)
    pred=f[["Date","SKU_ID","Region","P10","P50","P90"]].copy()
    pred["Date"]=pred.Date.dt.strftime("%Y-%m")
    pred["Units_Sold_Forecast"]=np.rint(pred.P50).astype(int)
    # Exact test-row order and key coverage
    test_keys=test[["Date","SKU_ID","Region","Product_Family","Business_Unit","Customer_Name","Sales_Channel"]].copy()
    test_keys["Date"]=pd.to_datetime(test_keys.Date).dt.strftime("%Y-%m")
    out=test_keys.merge(pred,on=["Date","SKU_ID","Region"],how="left")
    out["Units_Sold_Forecast"]=out.Units_Sold_Forecast.fillna(0).astype(int)
    last_price=train.sort_values("Date").groupby("SKU_ID").Unit_Price_Clean.last()
    out["Planning_Unit_Price_USD"]=out.SKU_ID.map(last_price)
    out["Forecast_Revenue_USD"]=out.Units_Sold_Forecast*out.Planning_Unit_Price_USD
    expected=test_keys[["Date","SKU_ID","Region"]].astype(str).reset_index(drop=True)
    actual=out[["Date","SKU_ID","Region"]].astype(str).reset_index(drop=True)
    if len(out)!=len(test) or not expected.equals(actual): raise RuntimeError("Forecast keys do not exactly match test.csv")
    if out[["P10","P50","P90"]].isna().any().any(): raise RuntimeError("Missing quantile forecast values")
    if ((out.P10>out.P50)|(out.P50>out.P90)).any(): raise RuntimeError("Quantile crossing detected")
    if (out.Units_Sold_Forecast<0).any(): raise RuntimeError("Negative submission forecast")
    out[["Date","SKU_ID","Region","Units_Sold_Forecast"]].to_csv(OUT/"submission.csv",index=False)
    out.to_csv(OUT/"forecast_with_intervals.csv",index=False)
    # Anomaly register with auditable reason/action fields
    anomalies=train.loc[train.Extreme_Demand_Flag|train.Stockout_Adjusted|train.Lifecycle_Conflict,
        ["Date","SKU_ID","Region","Product_Family","Customer_Name","Sales_Channel","Units_Sold","Demand_Unconstrained","Stockout_Adjusted","Lifecycle_Conflict","Extreme_Demand_Flag"]].copy()
    anomalies["Anomaly_Type"]=np.select([anomalies.Stockout_Adjusted,anomalies.Lifecycle_Conflict,anomalies.Extreme_Demand_Flag],
        ["Stockout / censored shipment","Lifecycle conflict","Extreme demand"],default="Review")
    anomalies["Action"]="Review against events calendar; stockout rows use requested units as demand"
    anomalies.to_csv(OUT/"anomaly_register.csv",index=False)
    # Optional planning analytics, built from pipeline outputs.
    from expert_outputs import build_outputs
    build_outputs(ROOT,OUT)
    # Summary
    rolling=pd.read_csv(OUT/"rolling_origin_19m_metrics.csv").iloc[0]
    summary={"train_rows_clean":int(len(train)),"test_rows":int(len(test)),"forecast_rows":int(len(out)),
      "missing_forecasts":int(out.Units_Sold_Forecast.isna().sum()),"negative_forecasts":int((out.Units_Sold_Forecast<0).sum()),"duplicate_submission_keys":int(out.duplicated(["Date","SKU_ID","Region"]).sum()),"test_key_order_exact":bool(expected.equals(actual)),"forecast_interval_rows":int(len(out)),"quantile_crossings":int(((out.P10>out.P50)|(out.P50>out.P90)).sum()),"forecast_total_19m":int(out.Units_Sold_Forecast.sum()),
      "backtest_wape_one_step":float(wape),"backtest_bias_one_step":float(bias),"backtest_wape_19m":float(rolling.WAPE),"backtest_bias_19m":float(rolling.Bias),"model":"Global LightGBM Tweedie validation plus LightGBM quantile-regression P10/P50/P90 forecasts; SKU-region lags, rolling statistics, event scope, lifecycle, replacement demand and macro lags",
      "interval_note":"P10/P50/P90 are LightGBM quantile-regression estimates; validate calibration by segment before operational use."}
    (OUT/"run_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))
if __name__=="__main__": main()
