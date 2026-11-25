from pathlib import Path
import pandas as pd, numpy as np, streamlit as st
import plotly.express as px
ROOT=Path(__file__).resolve().parent; OUT=ROOT/'outputs'
st.set_page_config(page_title='Celeris | Demand Control Tower',layout='wide')
st.title('Celeris Automobili Industrial')
st.caption('Demand Forecasting & Planning Control Tower | Historical demand + 19-month outlook')
@st.cache_data
def load():
    h=pd.read_csv(OUT/'cleaned_train.csv',parse_dates=['Date'])
    f=pd.read_csv(OUT/'forecast_with_intervals.csv')
    f['Date']=pd.to_datetime(f['Date'])
    q=pd.read_csv(OUT/'data_quality_report.csv')
    qp=pd.read_csv(OUT/'data_quality_profile.csv')
    a=pd.read_csv(OUT/'anomaly_register.csv',parse_dates=['Date'])
    b=pd.read_csv(OUT/'backtest_metrics.csv')
    r=pd.read_csv(OUT/'rolling_origin_19m_metrics.csv')
    seg=pd.read_csv(OUT/'backtest_segment_metrics.csv')
    ml=pd.read_csv(OUT/'macro_lead_lag_correlations.csv')
    pr=pd.read_csv(OUT/'procurement_plan_p90.csv')
    li=pd.read_csv(OUT/'lifecycle_forecast.csv')
    so=pd.read_csv(OUT/'stockout_lost_sales_summary.csv')
    ex=pd.read_csv(OUT/'planner_exceptions.csv')
    return h,f,q,qp,a,b,r,seg,ml,pr,li,so,ex
h,f,q,qp,a,b,r,seg,ml,pr,li,so,ex=load()
# Common filters shared by all views
st.sidebar.header('Global filters')
def opts(df,c): return sorted(df[c].dropna().astype(str).unique().tolist())
def filt(label,values): return st.sidebar.multiselect(label,values,default=values)
families=filt('Product family',opts(h,'Product_Family'))
skus=filt('SKU',opts(h,'SKU_ID'))
regions=filt('Region',opts(h,'Region'))
channels=filt('Sales channel',opts(h,'Sales_Channel'))
customers=filt('Customer',opts(h,'Customer_Name'))
def apply(df):
    z=df.copy()
    for c,v in [('Product_Family',families),('SKU_ID',skus),('Region',regions),('Sales_Channel',channels),('Customer_Name',customers)]:
        if c in z.columns and v: z=z[z[c].astype(str).isin(v)]
    return z
H=apply(h); F=apply(f)
last_price=H.sort_values('Date').groupby('SKU_ID').Unit_Price_Clean.last()
F['Planning_Unit_Price_USD']=F.SKU_ID.map(last_price)
F['Forecast_Revenue_USD']=F.Units_Sold_Forecast.fillna(0)*F.Planning_Unit_Price_USD
# Revenue history from cleaned unconstrained demand and clean unit prices
H['Revenue_Clean']=H['Demand_Unconstrained']*H['Unit_Price_Clean']
F['Units_Sold_Forecast']=F['Units_Sold_Forecast'].fillna(0)
# Shared timeline data
hist=H.groupby('Date',as_index=False).agg(Units=('Demand_Unconstrained','sum'),Revenue=('Revenue_Clean','sum'))
fc=F.groupby('Date',as_index=False).agg(Units=('Units_Sold_Forecast','sum'),Revenue=('Forecast_Revenue_USD','sum'))
hist['Part']='History'; fc['Part']='Forecast'
timeline=pd.concat([hist,fc],ignore_index=True)
tabs=st.tabs(['Executive overview','Data health','Demand explorer','Anomaly monitor','Forecast & accuracy','Planning extras'])
with tabs[0]:
    st.subheader('Executive overview')
    c1,c2,c3,c4=st.columns(4)
    c1.metric('19-month forecast units',f"{F.Units_Sold_Forecast.sum():,.0f}")
    c2.metric('Forecast revenue (USD)',f"${F.Forecast_Revenue_USD.sum():,.0f}")
    c3.metric('Historical demand units',f"{H.Demand_Unconstrained.sum():,.0f}")
    c4.metric('Filtered SKU-region rows',f"{len(F):,}")
    fig=px.line(timeline,x='Date',y='Units',color='Part',title='Monthly units: history into forecast')
    fig.add_vline(x=pd.Timestamp('2026-11-01'),line_dash='dash',annotation_text='Forecast starts')
    st.plotly_chart(fig,use_container_width=True)
    fig2=px.line(timeline,x='Date',y='Revenue',color='Part',title='Monthly revenue (USD; forecast uses last known SKU price)')
    fig2.add_vline(x=pd.Timestamp('2026-11-01'),line_dash='dash')
    st.plotly_chart(fig2,use_container_width=True)
    left,right=st.columns(2)
    with left: st.plotly_chart(px.bar(H.groupby('Product_Family',as_index=False).Demand_Unconstrained.sum(),x='Product_Family',y='Demand_Unconstrained',title='Historical demand mix by family'),use_container_width=True)
    with right: st.plotly_chart(px.bar(F.groupby('Region',as_index=False).Units_Sold_Forecast.sum(),x='Region',y='Units_Sold_Forecast',title='Forecast mix by region'),use_container_width=True)
    st.plotly_chart(px.bar(F.groupby('Sales_Channel',as_index=False).Units_Sold_Forecast.sum(),x='Sales_Channel',y='Units_Sold_Forecast',title='Forecast mix by channel'),use_container_width=True)
with tabs[1]:
    st.subheader('Data health')
    st.markdown('**Input profile: before vs after cleaning**')
    st.dataframe(qp,use_container_width=True,hide_index=True)
    st.markdown('**Issue register**')
    st.dataframe(q,use_container_width=True,hide_index=True)
    st.caption('Issue counts can overlap because one row may contain multiple quality issues.')
    st.metric('Clean training rows',f'{len(h):,}')
    st.metric('Rows in anomaly register',f'{len(a):,}')
with tabs[2]:
    st.subheader('Demand explorer')
    if H.empty: st.warning('No history matches the selected filters.')
    else:
        choices=sorted(H.SKU_ID.dropna().unique())
        chosen=st.selectbox('Inspect SKU',choices)
        z=H[H.SKU_ID==chosen]
        fig=px.line(z.groupby('Date',as_index=False).Demand_Unconstrained.sum(),x='Date',y='Demand_Unconstrained',title=f'{chosen}: unconstrained monthly demand')
        st.plotly_chart(fig,use_container_width=True)
        m=z.groupby('Date').agg(Units=('Demand_Unconstrained','sum'),ObservedRows=('observed','sum') if 'observed' in z.columns else ('Demand_Unconstrained','count')).reset_index()
        st.write('Zero-demand months:',int((m.Units==0).sum()),' | Months represented:',len(m))
        st.dataframe(z[['Date','SKU_ID','Region','Customer_Name','Sales_Channel','Units_Sold','Demand_Unconstrained','Stockout_Adjusted','Lifecycle_Conflict']].tail(100),use_container_width=True,hide_index=True)
with tabs[3]:
    st.subheader('Anomaly monitor')
    aa=apply(a)
    st.metric('Flagged records in current filter',f'{len(aa):,}')
    st.dataframe(aa.sort_values('Date',ascending=False),use_container_width=True,hide_index=True)
    if not aa.empty: st.plotly_chart(px.scatter(aa,x='Date',y='Demand_Unconstrained',color='Anomaly_Type',hover_data=['SKU_ID','Region','Action'],title='Flagged demand points'),use_container_width=True)
with tabs[4]:
    st.subheader('Forecast & accuracy')
    st.write('One-step validation and fixed-origin 19-month holdout:')
    st.dataframe(b,use_container_width=True,hide_index=True)
    st.dataframe(r,use_container_width=True,hide_index=True)
    st.markdown('**Backtest performance by family and region**')
    st.dataframe(seg,use_container_width=True,hide_index=True)
    st.caption('WAPE = sum(abs(actual − predicted)) / sum(actual). Bias = (forecast total − actual total) / actual total. The 19-month holdout freezes series history at its origin; it is a direct-horizon test, not 19 separate rolling origins.')
    st.plotly_chart(px.line(timeline,x='Date',y='Units',color='Part',title='Actual history vs P50 forecast'),use_container_width=True)
    fig=px.line(F.groupby('Date',as_index=False).agg(P10=('P10','sum'),P50=('P50','sum'),P90=('P90','sum')),x='Date',y=['P10','P50','P90'],title='Quantile forecast range')
    st.plotly_chart(fig,use_container_width=True)
    st.dataframe(F[['Date','SKU_ID','Region','Units_Sold_Forecast','P10','P50','P90']].head(500),use_container_width=True,hide_index=True)
with tabs[5]:
    st.subheader('Planning extras')
    left,right=st.columns(2)
    with left:
        st.markdown('**Lifecycle outlook**')
        st.dataframe(F.groupby('SKU_ID',as_index=False).agg(Forecast_Units=('Units_Sold_Forecast','sum')).sort_values('Forecast_Units',ascending=False).head(30),use_container_width=True,hide_index=True)
    with right:
        st.markdown('**Procurement planning proxy**')
        st.caption('P90 totals are a conservative planning proxy; confirm supplier lead times, MOQ and safety-stock policy before purchase orders.')
        st.metric('Filtered P90 units',f"{F.P90.sum():,.0f}")
    st.download_button('Download filtered forecast CSV',F.to_csv(index=False).encode(),file_name='filtered_forecast.csv',mime='text/csv')
    st.subheader('Macro lead/lag diagnostics')
    fams=sorted(ml.Product_Family.dropna().unique())
    mfam=st.selectbox('Family for macro correlations',fams,key='macro_family')
    mz=ml[ml.Product_Family==mfam].pivot(index='Indicator',columns='Demand_Lead_Months',values='Pearson_Correlation')
    st.plotly_chart(px.imshow(mz,aspect='auto',color_continuous_scale='RdBu',zmin=-1,zmax=1,title='Descriptive macro correlation by indicator lag'),use_container_width=True)
    st.caption('Correlations are exploratory associations, not causal elasticities.')
    left,right=st.columns(2)
    with left:
        st.markdown('**Stockouts and lost sales**')
        st.dataframe(so,use_container_width=True,hide_index=True)
    with right:
        st.markdown('**Lifecycle exposure**')
        st.dataframe(li.sort_values('Forecast_P90',ascending=False).head(30),use_container_width=True,hide_index=True)
    st.markdown('**P90 procurement plan**')
    PR=apply(pr)
    st.dataframe(PR.head(500),use_container_width=True,hide_index=True)
    st.markdown('**Planner exception queue**')
    EX=apply(ex)
    st.dataframe(EX.head(500),use_container_width=True,hide_index=True)
    st.download_button('Download procurement plan CSV',PR.to_csv(index=False).encode(),file_name='procurement_plan_p90.csv',mime='text/csv')
    st.download_button('Download planner exceptions CSV',EX.to_csv(index=False).encode(),file_name='planner_exceptions.csv',mime='text/csv')
    st.download_button('Download anomaly register CSV',aa.to_csv(index=False).encode(),file_name='anomaly_register.csv',mime='text/csv')
st.divider(); st.caption('Synthetic hackathon data. P10/P50/P90 are quantile-regression estimates; review segment calibration and planning constraints before operational use. Use the README for assumptions and limitations.')
