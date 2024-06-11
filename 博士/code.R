# 导入必须工具包
library(TwoSampleMR)
library(readr)
library(data.table)
library(ieugwasr)
library(MRPRESSO)
# 读取暴露结果数据
activity <- read.csv('activity_id.csv',header = F)
handgrip <- read.csv('hand grip_id.csv',header = F)
id_exposure <- fread("C:\\Users\\86182\\.spyder-py3\\mr_ieu_id\\Medication_id.csv")
# 获取暴露
a <- id_exposure[[1]]
d <- data.frame()
k <- data.frame()
for (v in a) {
  if(strsplit(v,'[-]')[[1]][1]!='ebi'){
    repeat{
      try({
        ## 提取暴露变量
        exposure_online <- extract_instruments(outcomes = "ukb-b-7385")
      })
      if(exists("exposure_online")){break}
      Sys.sleep(2)
    }
    outcome_online <- extract_outcome_data(snps = exposure_online$SNP,outcomes = 'ebi-a-GCST90014022')
    dat<-harmonise_data(exposure_dat = exposure_online,outcome_dat = outcome_online)
    write.csv(exposure_online,'D:\\怒放\\exposure_online.csv')
    write.csv(dat,'D:\\怒放\\dat.csv')
    mr <- mr(dat)
    d <- rbind(d,mr)
    k <- rbind(k,mr[mr$method=='Inverse variance weighted',])
  }
}
write.csv(d,'D:\\怒放\\d.csv')
write.csv(k,'D:\\怒放\\k.csv')
write.csv(mr,'mr.csv')
# 绘制散点图+回归图
mr_scatter_plot(mr,dat)
# 绘制森林图
mr_forest_plot(mr_singlesnp(dat))
# 分析异质性
mr_heterogeneity(dat)
write.csv(mr_heterogeneity(dat),'D:\\怒放\\heterogeneity.csv')
# 绘制漏斗图——异质性可视化
mr_funnel_plot(singlesnp_results = mr_singlesnp(dat))
# 分析多效性
mr_pleiotropy_test(dat) 
write.csv(mr_pleiotropy_test(dat),'D:\\怒放\\pleiotropy.csv')
# 敏感性检验
mr_leaveoneout_plot(leaveoneout_results = mr_leaveoneout(dat))
presso <- mr_presso(BetaExposure = 'beta.exposure',BetaOutcome = 'beta.outcome',SdExposure = 'se.exposure',SdOutcome = 'se.outcome',dat,OUTLIERtest = T, DISTORTIONtest = T)
write.csv(presso,'D:\\怒放\\presso.csv')

d <- read.csv('D:\\怒放\\d.csv')
d <- generate_odds_ratios(d)
View(d)
