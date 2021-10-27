library(ggplot2)
library(scales)
library(ggpubr)
library(dict)
source("theme.R")

line_size <- 1.5
alpha_line <- 0.75
point_size_hline <- 0.3

fig_width       <- 7
fig_height      <- 5

basesize <- 15


unfairness_range_plot <- function(dataset, rseed, model_class, title) {
    
    input_file              <- sprintf("../results/%s_%s_%s.csv", dataset, model_class, rseed)
    input_file_baseline     <- sprintf("./unfairness_bbox/%s.csv", dataset)

    save_path <- "./graphs/unfairness_range"
    
    dir.create(save_path, showWarnings = FALSE, recursive = TRUE)

    output_file <- sprintf("%s/%s_%s_%s.png", save_path, dataset, model_class, rseed)

    df                      <- read.csv(input_file, header=T)
    df_baseline             <- read.csv(input_file_baseline, header=T)

    pp <- ggplot(df, aes(x = factor(round(fidelity_explainer, 4)))) + 


    geom_errorbar(aes(ymin = min_disp, ymax = max_disp), colour = "#0e5357", size=line_size, width = 0.25) + 
    geom_hline(data = df_baseline, mapping = aes(yintercept = unfairness), size=point_size_hline, linetype = "dashed") +


    theme_light_v2(base_size=basesize) + 

    scale_color_manual(
        name="Group of interest",
        labels=c("Suing group", "Test set"),
        values=c("#0e5357", "#cda815")
    )  +

    
    labs(x = "Fidelity", y = "Unfairness", title = title) + 


    theme(
        legend.direction = "horizontal", 
        legend.position = "top", 
        legend.box = "horizontal") 

    


    pp 

    ggsave(output_file, dpi=300, width=fig_width, height=fig_height)
                    
}


# dataset map
dataset_map <- dict()
dataset_map[["adult_income"]]   <- "Adult Income"
dataset_map[["compas"]]         <- "COMPAS"
dataset_map[["default_credit"]] <- "Default Credit"
dataset_map[["marketing"]]      <- "Marketing"


#datasets <- c("adult_income", "compas", "default_credit", "marketing")
#model_classes <- c("AdaBoost", "DNN", "RF", "XgBoost")

datasets <- c("adult_income", "compas", "default_credit", "marketing")
model_classes <- c("DNN")



for (dataset in datasets){
    for (model_class in model_classes){
            unfairness_range_plot(dataset, 0, model_class, sprintf("%s -- %s", dataset_map[[dataset]], model_class))
    }
}
