library(tidyverse)
library(DataExplorer)
library(dplyr)
library(ggplot2)

setwd("C:/LocalData/pabflore/encoder-pacman")

df <- read_csv("data/psych/AiPerCogPacman_DATA_2025-04-01_1343.csv") %>% dplyr::filter(record_id > 60)



# Data Cleaning (and removing myself)
consent_bisbas <- df %>% dplyr::filter(!is.na(consent_timestamp) & record_id > 60) %>%  dplyr::select(c(1,5:35)) %>% drop_columns(c("name", "email"))

bisbas <- consent_bisbas %>% dplyr::select(c(1,10:29))

bisbas <- bisbas %>% mutate(BIS = bis_1 + bis_2 + bis_3 + bis_4 + bis_5 + bis_6 + bis_7,
                            DRIVE = drive_1 + drive_2 + drive_3 + drive_4,
                            REWARD = rew_1 + rew_2 + rew_3 + rew_4 + rew_5,
                            FUN = fun_1 + fun_2 + fun_3 + fun_4)




flow <- df %>% dplyr::filter(redcap_repeat_instrument == "flow" & record_id > 60) %>% dplyr::select(c(1,'redcap_repeat_instance',47:54)) %>% mutate(FLOW = (fss_1 + fss_2 + fss_3 + fss_4 + fss_5 + fss_6 + fss_7 + fss_8)/8)



merged <- bisbas %>% left_join(flow, by = "record_id")


# flow_series <- flow[c(1,2,11)] %>% mutate(record_id = as.factor(record_id)) %>% spread(key = redcap_repeat_instance, value = FLOW) %>% mutate(record_id = as.numeric(record_id))

flow_series <- flow[c(1,2,11)] %>% mutate(record_id = as.factor(record_id))


# Filter out record_ids that have less than 5 repeating instances
flow_series <- flow_series %>% group_by(record_id) %>% filter(n() >= 10) %>% ungroup()
  
flow_series %>% ggplot(aes(x = redcap_repeat_instance, y = FLOW, colour = record_id)) + 
  geom_smooth(method = lm) + 
  geom_point() + 
  labs(title = "Flow Series", x = "Instance", y = "Flow Score")




# Data Exploration
create_report(bisbas, output_file = "bisbas_report.html")
create_report(flow, output_file = "flow_report.html")
create_report(merged, output_file = "merged_report.html")
