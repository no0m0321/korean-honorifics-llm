# T1-1: H4 상호작용을 lme4::glmer로 재추정 (사전등록 1순위 모형의 정본 적합).
# correct ~ level*model + (1|item_id) + (1|template)  — item과 template를 교차 무선효과로 분리.
# GEE가 둘을 단일 교환가능 상관으로 뭉갠 것을 교정. 상호작용 동시검정은 우도비검정(LRT).
#
# 실행: Rscript analysis/h4_glmer.R

suppressMessages({ library(lme4) })

run_task <- function(task) {
  d <- read.csv(sprintf("analysis/output/glmm_input_%s.csv", task))
  d$level    <- factor(d$level)
  d$model    <- factor(d$model)
  d$item_id  <- factor(d$item_id)
  d$template <- factor(d$template)
  ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

  full <- glmer(correct ~ level * model + (1|item_id) + (1|template),
                data = d, family = binomial, control = ctrl, nAGQ = 0)
  red  <- glmer(correct ~ level + model + (1|item_id) + (1|template),
                data = d, family = binomial, control = ctrl, nAGQ = 0)
  lrt <- anova(red, full)             # 상호작용 동시 우도비검정
  chi2 <- lrt$Chisq[2]; df <- lrt$Df[2]; p <- lrt$`Pr(>Chisq)`[2]
  sing <- isSingular(full)

  cat(sprintf("\n== %s ==\n", toupper(task)))
  cat(sprintf("  glmer LRT(상호작용): chi2(%d)=%.1f, p=%.3g, singular=%s\n",
              df, chi2, p, sing))
  vc <- as.data.frame(VarCorr(full))
  cat(sprintf("  무선효과 분산: item=%.4f, template=%.4f\n",
              vc$vcov[vc$grp=="item_id"], vc$vcov[vc$grp=="template"]))
  data.frame(task=task, method="glmer (1|item)+(1|template)",
             chi2=round(chi2,1), df=df, p=p, singular=sing,
             var_item=round(vc$vcov[vc$grp=="item_id"],4),
             var_template=round(vc$vcov[vc$grp=="template"],4))
}

res <- rbind(run_task("nsmc"), run_task("copa"))
write.csv(res, "analysis/output/h4_glmer.csv", row.names = FALSE)
cat("\n저장: analysis/output/h4_glmer.csv\n")
