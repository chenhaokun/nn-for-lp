#coding:utf-8
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdate
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import os


mpl.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
mpl.rcParams['axes.unicode_minus']=False #用来正常显示负号
mpl.rc('xtick', labelsize=20) #设置坐标轴刻度显示大小
mpl.rc('ytick', labelsize=20)
font_size=30
#matplotlib.rcParams.update({'font.size': 60})

# %matplotlib inline
plt.style.use('ggplot')

data=pd.read_csv('simsendLogConvert_20160803094801.csv',index_col=0,encoding='gb2312',parse_dates=True)

columns_len=len(data.columns)
data_columns=data.columns

for x in range(0,columns_len,2):
    print('第{}列'.format(x))
    total=data.ix[:,x]
    print('第{}列'.format(x+1))
    successRate=(data.ix[:,x+1]/data.ix[:,x]).fillna(0)


    yLeftLabel=data_columns[x]
    yRightLable=data_columns[x+1]


    print('------------------开始绘制类型{}曲线图------------------'.format(data_columns[x]))

    fig=plt.figure(figsize=(25,20))
    ax1=fig.add_subplot(111)
    #绘制Total曲线图
    ax1.plot(total,color='#4A7EBB',label=yLeftLabel,linewidth=4)

    # 设置X轴的坐标刻度线显示间隔
    ax1.xaxis.set_major_formatter(mdate.DateFormatter('%Y-%m-%d %H:%M:%S'))#设置时间标签显示格式
    plt.xticks(pd.date_range(data.index[0],data.index[-1],freq='1min'))#时间间隔
    plt.xticks(rotation=90)

    #设置双坐标轴，右侧Y轴
    ax2=ax1.twinx()

    #设置右侧Y轴显示百分数
    fmt='%.2f%%'
    yticks = mtick.FormatStrFormatter(fmt)

    # 绘制成功率图像
    ax2.set_ylim(0,110)
    ax2.plot(successRate*100,color='#BE4B48',label=yRightLable,linewidth=4)
    ax2.yaxis.set_major_formatter(yticks)

    ax1.set_xlabel('Time',fontsize=font_size)
    ax1.set_ylabel(yLeftLabel,fontsize=font_size)
    ax2.set_ylabel(yRightLable,fontsize=font_size)

    legend1=ax1.legend(loc=(.02,.94),fontsize=16,shadow=True)
    legend2=ax2.legend(loc=(.02,.9),fontsize=16,shadow=True)

    legend1.get_frame().set_facecolor('#FFFFFF')
    legend2.get_frame().set_facecolor('#FFFFFF')

    plt.title(yLeftLabel+'&'+yRightLable,fontsize=font_size)

    plt.savefig('D:\\JGT\\Work-YL\\01布置的任务\\04绘制曲线图和报告文件\\0803\\出图\\{}-{}'.format(yLeftLabel.replace(r'/',' '),yRightLable.replace(r'/',' ')),dpi=300)
