# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 14:53:06 2018

@author: linzhaoshu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May  4 15:30:34 2018

@author: linzhaoshu
"""

from sklearn.decomposition import PCA
import pandas as pd
from sklearn import linear_model
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import sys




def ycm(to, x):
    return x[0]+x[1]*((1-np.exp(-to/x[4]))/(to/x[4]))+x[2]*(((1-np.exp(-to/x[4]))/(to/x[4]))-np.exp(-to/x[4]))+x[3]*(((1-np.exp(-to/x[5]))/(to/x[5]))-np.exp(-to/x[5]))



def ACMresultfunc(startdate,enddate,CurveParam,CurveEndTerm,factorKnum,ShowTerms):
#startdate='1970-01-31'
#enddate='2018-06-30'
#找到月末日期。。。
    dates=list(CurveParam.index)
    b=[a.split('-') for a in dates]
    cpan=pd.DataFrame([[bm[0]+'-'+bm[1],bm[2]] for bm in b],columns=['M','d'])
    x=cpan.groupby('M').max()
    monthenddates=[a+'-'+b for a,b in zip(list(x.index),list(x['d']))]
    #ACMdata.index=monthenddates[:-1]
   
    monthenddates=monthenddates[monthenddates.index(startdate):monthenddates.index(enddate)+1]
    CurveParamMonthEnd=CurveParam.loc[monthenddates,:]
   
    
    
    ############################
    #根据GSW的Curve计算出完整Curve（因为GSW对于很老的Curve不提供超过7年的点）
   
    Terms=[float(a)/12 for a in range(1,CurveEndTerm)]
    Termname=[str(a)+'M' for a in range(1,CurveEndTerm)]
    termtemp=[]
    for monthenddate in monthenddates:
        B0,B1,B2,B3,T1,T2=CurveParamMonthEnd.loc[monthenddate,:]
        temp=[]
    #    for Term in Terms:
    #        temp.append(B0+B1*(1-np.exp(-Term/T1))/(Term/T1)+B2*((1-np.exp(-Term/T1))/(Term/T1)-np.exp(-Term/T1))+B3*((1-np.exp(-Term/T2))/(Term/T2)-np.exp(-Term/T2)))
    #    termtemp.append(temp)
        termtemp.append(ycm(np.array(Terms),np.array([B0,B1,B2,B3,T1,T2])))
    GSW_Curve=pd.DataFrame(termtemp,columns=Termname,index=monthenddates)/1200
   
    #############################
   
    #得到主成分
    #GSW_CurveforPCA=GSW_Curve[Termname[2::3]]
    GSW_CurveforPCA=GSW_Curve
    GSW_Curve_std = StandardScaler().fit_transform(GSW_CurveforPCA)
    sklearn_pca = PCA(n_components=factorKnum)
    PCAout = sklearn_pca.fit_transform(GSW_Curve_std)
    PCA5factors=pd.DataFrame(PCAout,columns=[str(a)+' Factor' for a in range(1,factorKnum+1)])
    PCA5factors.index=monthenddates
    #对主成分进行自回归
    reg = linear_model.LinearRegression()
    PCA5factorst=PCA5factors[1:]#factor t
    PCA5factorst_1=PCA5factors[:-1]#factor t-1
    reg.fit (PCA5factorst_1, PCA5factorst)
    v=PCA5factorst-reg.predict(PCA5factorst_1)#residual
    v_sigma=v.T.dot(v)/len(monthenddates)#residual vol 用于后面的回归
    PHI=reg.coef_
    MU=reg.intercept_
    #############################
    #计算超额收益
    ExcessRt=pd.DataFrame([],columns=Terms)
    EXRttemp=[]
    i=0#记录monthenddate的位置
    for monthenddate in monthenddates[:-1]:
        temp=[]
        j=0
        for Term in Terms:
            if j==0:
                temp.append(0)
            else:
                temp.append(GSW_Curve.iloc[i,j]*Terms[j]*12-GSW_Curve.iloc[i+1,j-1]*Terms[j-1]*12-GSW_Curve.iloc[i,0])
            j=j+1
        EXRttemp.append(temp)   
        i=i+1
    ExcessRt=pd.DataFrame(EXRttemp,columns=Termname,index=monthenddates[:-1])
   
    #############################
    #
    ExcessRtreg = linear_model.LinearRegression()
    ExcessRtreg.fit(np.hstack((v.values,PCA5factorst_1.values)),ExcessRt)
   
    #
    #rx=np.mat(ExcessRt).T
    #zz=np.mat(np.hstack((v.values,PCA5factorst_1.values))).T
    #a=rx*zz.T*np.linalg.inv(zz*zz.T)
   
    ExcessRtv=ExcessRt-ExcessRtreg.predict(np.hstack((v.values,PCA5factorst_1.values)))
    ExcessRtv_sigmatrace=np.trace(ExcessRtv.T.dot(ExcessRtv))/(CurveEndTerm-1)/len(monthenddates)
    beta=ExcessRtreg.coef_[:,:factorKnum].T
    c=ExcessRtreg.coef_[:,factorKnum:].T
    Betastar=[a.dot(v_sigma).dot(a) for a in beta.T]
    atemp=(ExcessRtv_sigmatrace+Betastar)/2+ExcessRtreg.intercept_
   
    ##
    lambda0reg = linear_model.LinearRegression()
    lambda0reg.fit_intercept=False
    lambda0reg.fit(beta.T,atemp)
    lambda0=lambda0reg.coef_
   
    ExcessRtv_vec=np.mat(ExcessRtv).T
    ExcessRtv_sigmatrace_vec=np.trace(ExcessRtv_vec*ExcessRtv_vec.T)/len(monthenddates)/(CurveEndTerm-1)
    ExRt_a=np.mat(ExcessRtreg.intercept_).T
   
    Betastar_frombeta=[]
    betamat=np.mat(beta)
    for betai in betamat.T:
        vecbeta=np.hstack(betai.T*betai)
        Betastar_frombeta.append(np.matrix.tolist(vecbeta)[0])
    Betastar_frombeta=np.mat(Betastar_frombeta)
    vec_vsigma=np.hstack(np.mat(v_sigma)).T
    avec=(Betastar_frombeta*vec_vsigma+ExcessRtv_sigmatrace)/2+ExRt_a
   
    lambda0vec=np.linalg.inv(betamat*betamat.T)*betamat*avec
   
    ##
    lambda1reg = linear_model.LinearRegression()
    lambda1reg.fit_intercept=False
    lambda1reg.fit(beta.T,c.T)
    lambda1=lambda1reg.coef_.T
    ##
    lambda1vec=np.linalg.inv(betamat*betamat.T)*betamat*c.T
   
    #############################
   
    #
   #APredict=[0]
    #BPredict=[0]
    deltareg = linear_model.LinearRegression()
    deltareg.fit(PCA5factors,GSW_Curve['1M'])
    #APredict.append(-deltareg.intercept_)
    #BPredict.append(-deltareg.coef_)
    #
    #for i in range(2,CurveEndTerm):
    #    BPredict.append(BPredict[i-1].dot(PHI-lambda1)-deltareg.coef_)
    #    APredict.append(APredict[i-1]+BPredict[i-1].T.dot(MU-lambda0)+1/2*(BPredict[i-1].dot(v_sigma).dot(BPredict[i-1])+ExcessRtv_sigmatrace)-deltareg.intercept_)
    #
    #a=(APredict[120]+BPredict[120].dot(PCA5factors.T))
    ############################
   
    
    FittedYield=pd.DataFrame([],index=monthenddates)
    for showterm in ShowTerms:
        showtermname=str(int(showterm/12))+'Y'
        APredictvec=[0]
        BPredictvec=[np.mat(np.zeros(factorKnum)).T]
        delta0=deltareg.intercept_
        delta1=np.mat(deltareg.coef_).T
        APredictvec.append(-delta0)
        BPredictvec.append(-delta1)
        for i in range(2,CurveEndTerm):
            BPredictvec.append((BPredictvec[i-1].T*(np.mat(PHI)-lambda1vec)-delta1.T).T)
            APredictvec.append(np.float(APredictvec[i-1]+BPredictvec[i-1].T*(np.mat(MU).T-lambda0vec)+1/2*(BPredictvec[i-1].T*np.mat(v_sigma)*BPredictvec[i-1]+ExcessRtv_sigmatrace)-delta0))
        avvv=APredictvec[showterm]+BPredictvec[showterm].T*np.mat(PCA5factors.values).T
       
        
        APredictvecRF=[0]
        BPredictvecRF=[np.mat(np.zeros(factorKnum)).T]
        APredictvecRF.append(-delta0)
        BPredictvecRF.append(-delta1)
        for i in range(2,CurveEndTerm):
            BPredictvecRF.append((BPredictvecRF[i-1].T*(np.mat(PHI))-delta1.T).T)
            APredictvecRF.append(np.float(APredictvecRF[i-1]+BPredictvecRF[i-1].T*(np.mat(MU).T)+1/2*(BPredictvecRF[i-1].T*np.mat(v_sigma)*BPredictvecRF[i-1]+ExcessRtv_sigmatrace)-delta0))
        avvvRF=APredictvecRF[showterm]+BPredictvecRF[showterm].T*np.mat(PCA5factors.values).T
       
        avvvpd=pd.DataFrame(np.matrix.tolist(-avvv/showterm*1200)[0])
        avvvRFpd=pd.DataFrame(np.matrix.tolist(-avvvRF/showterm*1200)[0])
        FittedYield[showtermname+' Yield']=avvvpd.values
        FittedYield[showtermname+' RFR']=avvvRFpd.values
        FittedYield[showtermname+' TP']=FittedYield[showtermname+' Yield']-FittedYield[showtermname+' RFR']
    #res=pd.concat([FittedYield,ACMdata.loc[monthenddates,['ACMY01','ACMY02','ACMY05','ACMY10','ACMTP01','ACMTP02','ACMTP05','ACMTP10','ACMRNY01','ACMRNY02','ACMRNY05','ACMRNY10']]],axis=1)
    return FittedYield
#############################
#APredictRF=[0]
#BPredictRF=[0]
#deltareg = linear_model.LinearRegression()
#deltareg.fit(PCA5factors,GSW_Curve['1M'])
#APredictRF.append(deltareg.intercept_)
#BPredictRF.append(deltareg.coef_)
#
#for i in range(2,CurveEndTerm):
#    BPredictRF.append(BPredictRF[i-1].dot((PHI))-deltareg.coef_)
#    APredictRF.append(APredictRF[i-1]+BPredictRF[i-1].dot(MU)+1/2*(BPredictRF[i-1].dot(v_sigma).dot(BPredictRF[i-1])+ExcessRtv_sigmatrace)-deltareg.intercept_)
#
#aRF=(APredictRF[120]+BPredictRF[120].dot(PCA5factors.T))/120
#a-aRF
#fig=plt.figure(figsize=(10,7))
#ax1=fig.add_subplot(111)
#ax1.plot(avvv.T,color='#4A7EBB')
#ax2=ax1.twinx()
#ax2.plot(aRF,color='#BE4B48')

CurveParam=pd.read_excel('GSW_CurveParam.xls','GSWParam',index_col=0)

#CurveParam=pd.read_excel('.\DE\DE_CurveParam.xls','GSWParam',index_col=0)
#GSWdataupto7=GSWdata[GSWdata.columns[0:7]].dropna()
#ACMdata=pd.read_excel('ACMTermPremium.xls','ACM Monthly',index_col=0)

CurveEndTerm=121
factorKnum=6
#startdate='1970-01-31'
#startdate='2008-03-31'
#startdate='1998-03-31'
#startdate='1988-03-31'
#startdate='1981-03-31'
#startdate='2013-03-28'
#startdate='1972-09-29'



#startdate='2007-12-31'
#enddate='2018-03-29'


#startdate='2009-06-30'

#enddate='2018-03-29'
#enddate='2008-03-31'
#enddate='1998-03-31'

#April 1960    February 1961
#December 1969 November 1970
#November 1973 March 1975
startdate='1961-06-30'
#startdate='1972-09-30'

#January 1980  July 1980
#周期过短，合并处理
#startdate='1975-03-31'
#enddate='1982-11-30'


#July 1981 November 1982
#startdate='1981-07-31'
#enddate='1990-07-31'
#startdate='1970-01-31'
#enddate='1979-06-30'



#July 1990 March 1991
#startdate='1979-06-30'
#enddate='1990-05-31'


#March 2001    November 2001
#startdate='1990-05-31'
#enddate='2008-05-31'


#December 2007 June 2009

#startdate='1986-03-31'
# enddate='2019-08-09'
enddate='2018-03-29'
ShowTerms=[12,24,60,120]


dates=list(CurveParam.index)
b=[a.split('-') for a in dates]
cpan=pd.DataFrame([[bm[0]+'-'+bm[1],bm[2]] for bm in b],columns=['M','d'])
x=cpan.groupby('M').max()
monthenddates=[a+'-'+b for a,b in zip(list(x.index),list(x['d']))]
#ACMdata.index=monthenddates[:-1]

monthenddates=monthenddates[monthenddates.index(startdate):monthenddates.index(enddate)+1]
movingwindow=240

Emptyvalue=pd.DataFrame([],index=monthenddates)
denominator=np.concatenate((np.arange(1,movingwindow+1),np.full(len(monthenddates)-movingwindow*2,movingwindow),np.arange(movingwindow,0,-1)))
for j in np.arange(0,len(monthenddates)-movingwindow):
#for j in np.arange(0,2):
#    if i <= movingwindow:
#        windowstart=1
#        windowend=i
#    elif i > len(monthenddates)-movingwindow:
#        windowstart=i-movingwindow
#        windowend=len(monthenddates)-movingwindow
#    else:
#        windowstart=i-movingwindow
#        windowend=i+movingwindow
#    for j in np.arange(windowstart,windowend+1):
    if j==0 :
        ACMresult=ACMresultfunc(monthenddates[j],monthenddates[j+movingwindow],CurveParam,CurveEndTerm,factorKnum,ShowTerms)
        ACMresultlongtemp=Emptyvalue.join(ACMresult).fillna(0)
        ACMresultlong=ACMresultlongtemp
    else:
        ACMresult=ACMresultfunc(monthenddates[j],monthenddates[j+movingwindow],CurveParam,CurveEndTerm,factorKnum,ShowTerms)
        ACMresultlongtemp=Emptyvalue.join(ACMresult).fillna(0)
        ACMresultlong=ACMresultlong+ACMresultlongtemp
    sys.stdout.write('{0}/\r'.format(j))
    sys.stdout.flush()
for j in np.arange(0,len(monthenddates)):
    ACMresultlong.iloc[j,:]=ACMresultlong.iloc[j,:]/denominator[j]

ACMresultlong.to_csv('us.csv')
