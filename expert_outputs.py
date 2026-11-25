"""Generate auditable optional planning views from the pipeline outputs."""
from pathlib import Path
import numpy as np
import pandas as pd

def build_outputs(root: Path, out: Path):
    data=root/'data'
    hist=pd.read_csv(out/'cleaned_train.csv',parse_dates=['Date'])
    fc=pd.read_csv(out/'forecast_with_intervals.csv')
    fc['Date']=pd.to_datetime(fc['Date'])
    master=pd.read_csv(data/'sku_master.csv')
    stock=pd.read_csv(data/'inventory_stockouts.csv')
    macro=pd.read_csv(out/'cleaned_macro.csv')
    macro['Date']=pd.to_datetime(macro['Date'])
    # Ordering-cycle and intermittency diagnostics; missing ERP rows are not recoded as zeros.
    rows=[]
    for (sku,region),g in hist.groupby(['SKU_ID','Region'],sort=False):
        g=g.sort_values('Date'); obs=g.Demand_Unconstrained.dropna(); nz=g.loc[g.Demand_Unconstrained>0,'Date'].sort_values()
        adi=len(obs)/max(int((obs>0).sum()),1)
        cv2=(obs.std(ddof=1)/obs.mean())**2 if len(obs)>1 and obs.mean()>0 else np.nan
        gaps=nz.diff().dt.days.div(30.4375).dropna()
        rows.append({'SKU_ID':sku,'Region':region,'Observed_Months':len(obs),'Nonzero_Months':int((obs>0).sum()),'ADI':adi,'CV2':cv2,'Median_Order_Gap_Months':float(gaps.median()) if len(gaps) else np.nan,'Intermittency_Class':'No observed demand' if not (obs>0).any() else ('Smooth' if adi<=1.32 and cv2<=0.49 else ('Intermittent' if adi>1.32 and cv2<=0.49 else 'Lumpy'))})
    pd.DataFrame(rows).to_csv(out/'ordering_cycle_diagnostics.csv',index=False)
    # Macro lead/lag correlations by product family. Descriptive, not causal.
    family=hist.groupby(['Date','Product_Family'],as_index=False).Demand_Unconstrained.sum()
    mcols=[c for c in macro.columns if c not in ['Date','Year','Month']]
    corr=[]
    for fam,g in family.groupby('Product_Family'):
        g=g.sort_values('Date')
        for ind in mcols:
            for lag in range(4):
                mm=macro[['Date',ind]].copy(); mm['Date']=mm.Date+pd.DateOffset(months=lag)
                z=g.merge(mm,on='Date').dropna()
                corr.append({'Product_Family':fam,'Indicator':ind,'Demand_Lead_Months':lag,'Pearson_Correlation':z.Demand_Unconstrained.corr(z[ind]) if len(z)>2 else np.nan,'Observations':len(z)})
    pd.DataFrame(corr).to_csv(out/'macro_lead_lag_correlations.csv',index=False)
    # P90 procurement proxy at monthly SKU-region grain.
    fc[['Date','SKU_ID','Region','Product_Family','Sales_Channel','Customer_Name','P50','P90']].assign(Date=lambda x:x.Date.dt.strftime('%Y-%m')).to_csv(out/'procurement_plan_p90.csv',index=False)
    # Lifecycle forecast exposure.
    life=fc.merge(master[['SKU_ID','Lifecycle_Status','Launch_Date','EOL_Date','Replaces_SKU']],on='SKU_ID',how='left')
    life.groupby(['Lifecycle_Status','SKU_ID','Replaces_SKU'],dropna=False,as_index=False).agg(Forecast_P50=('P50','sum'),Forecast_P90=('P90','sum')).to_csv(out/'lifecycle_forecast.csv',index=False)
    # Stockout/lost-sales evidence; do not add backorders to lost units.
    stock['Lost_Units']=pd.to_numeric(stock['Lost_Units'],errors='coerce').fillna(0)
    stock['Requested_Units']=pd.to_numeric(stock['Requested_Units'],errors='coerce').fillna(0)
    stock['Shipped_Units']=pd.to_numeric(stock['Shipped_Units'],errors='coerce').fillna(0)
    stock.groupby('Reason',dropna=False,as_index=False).agg(Stockout_Records=('SKU_ID','size'),Requested_Units=('Requested_Units','sum'),Shipped_Units=('Shipped_Units','sum'),Lost_Units=('Lost_Units','sum')).to_csv(out/'stockout_lost_sales_summary.csv',index=False)
    # Exceptions for planners: high forecast uncertainty, no median forecast, or announced EOL.
    exc=fc.merge(master[['SKU_ID','Lifecycle_Status','EOL_Date']],on='SKU_ID',how='left')
    exc['Interval_Width']=exc.P90-exc.P10
    exc['Exception_Reason']=np.select([exc.P50.eq(0),exc.Lifecycle_Status.eq('EOL Announced'),exc.Interval_Width.gt(exc.P50)],['Zero median forecast','EOL announced','Wide forecast interval'],default='Review')
    exc[exc.Exception_Reason.ne('Review')][['Date','SKU_ID','Region','Product_Family','Sales_Channel','Customer_Name','P10','P50','P90','Lifecycle_Status','EOL_Date','Exception_Reason']].to_csv(out/'planner_exceptions.csv',index=False)
    # Backtest metrics by family and region, plus the aggregate 19-month horizon result.
    bp=pd.read_csv(out/'backtest_predictions.csv',parse_dates=['Date'])
    meta=hist[['Date','SKU_ID','Region','Product_Family']].drop_duplicates(['Date','SKU_ID','Region'])
    bp=bp.merge(meta,on=['Date','SKU_ID','Region'],how='left')
    rows=[]
    for dimension in ['Product_Family','Region']:
        for name,g in bp.groupby(dimension,dropna=False):
            actual=g.Demand_Unconstrained.sum();
            for model_col,model_name in [('prediction','LightGBM'),('baseline_seasonal_naive','Seasonal naive'),('baseline_recent_mean','Recent mean')]:
                rows.append({'Horizon':'1-month one-step','Dimension':dimension,'Segment':name,'Model':model_name,'WAPE':np.abs(g[model_col]-g.Demand_Unconstrained).sum()/max(actual,1),'Bias':(g[model_col].sum()-actual)/max(actual,1),'Rows':len(g)})
    r=pd.read_csv(out/'rolling_origin_19m_metrics.csv').iloc[0]
    rows.append({'Horizon':'19-month fixed-origin','Dimension':'Overall','Segment':'All','Model':'LightGBM','WAPE':r.WAPE,'Bias':r.Bias,'Rows':r.rows})
    pd.DataFrame(rows).to_csv(out/'backtest_segment_metrics.csv',index=False)
    # Hierarchy totals: each level is a straight sum of the same bottom-level forecast.
    for keys,name in [(['Date'],'forecast_by_month.csv'),(['Date','Product_Family'],'forecast_by_family.csv'),(['Date','Region'],'forecast_by_region.csv'),(['Date','Sales_Channel'],'forecast_by_channel.csv')]:
        cols=[c for c in keys if c in fc.columns]
        if 'Sales_Channel' in keys and 'Sales_Channel' not in fc.columns: continue
        fc.groupby(cols,as_index=False).agg(P50=('P50','sum'),P90=('P90','sum')).to_csv(out/name,index=False)
