library(tidyverse)
library(DataExplorer)
library(dplyr)
library(ggplot2)

df <- read_csv("data/psych/psych.csv")


consent_bisbas <- df %>% dplyr::filter(!is.na(consent_timestamp) & record_id > 60) %>%  dplyr::select(c(5:35)) %>% drop_columns(c("name", "email"))

bisbas <- consent_bisbas %>% dplyr::select(c(9:28))

bisbas <- bisbas %>% mutate(BIS = bis_1 + bis_2 + bis_3 + bis_4 + bis_5 + bis_6 + bis_7,
                            DRIVE = drive_1 + drive_2 + drive_3 + drive_4,
                            REWARD = rew_1 + rew_2 + rew_3 + rew_4 + rew_5,
                            FUN = fun_1 + fun_2 + fun_3 + fun_4)




flow <- df %>% dplyr::filter(redcap_repeat_instrument == "flow" & record_id > 60) %>% dplyr::select(c(47:54)) %>% mutate(FLOW = fss_1 + fss_2 + fss_3 + fss_4 + fss_5 + fss_6 + fss_7 + fss_8)





# Data Exploration
create_report(bisbas, output_file = "bisbas_report.html")
create_report(flow, output_file = "flow_report.html")
